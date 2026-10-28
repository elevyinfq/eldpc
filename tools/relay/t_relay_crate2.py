#!/usr/bin/env python3
"""relay-bp 0.2.2 against the GPU library's Relay-BP, in two phases.

The CPU reference is slow (0.5-1.4 s per shot), so the comparison is split so the crate phase can
run in parallel processes without occupying the GPU:

  cpu  --problem P --seed S --shots N --s 1,5 --out F.npz
       fresh stim shots (the seed is printed and stored), relay-bp[stim]==0.2.2 RelayDecoderF64 at
       qLDPC's defaults (gamma0 0.1, pre_iter 80, num_sets 300, set_max_iter 60, stop_nconv = S) and
       the gross code's [-0.24, 0.66] interval; stores the detectors, the true observables, and the
       crate's per-shot observables / success / iterations for each S.

  gpu  --lib L --problem P --in F.npz --s 1,5 --out R.json [--relay-seed N]
       runs the GPU library on exactly those shots (config relay=S, same parameters), compares per shot,
       and appends one JSON row per S with each LER and its 99 % Clopper-Pearson interval, the
       number of failures and convergences, and a two-proportion z-test.  The row also carries the
       per-shot agreement (both fail / only ours / only crate) and a McNemar exact test on the
       discordant pairs.

Run with a Python environment that has relay-bp[stim]==0.2.2 installed: python -I t_relay_crate2.py ...
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "python"))
from dbench.formats import load_problem_dir                   # noqa: E402


def cp(k, n, alpha=0.01):
    from scipy.stats import beta
    if n <= 0:
        return [0.0, 1.0]
    lo = 0.0 if k == 0 else float(beta.ppf(alpha / 2.0, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1.0 - alpha / 2.0, k + 1, n - k))
    return [lo, hi]


def two_prop(k1, n1, k2, n2):
    from scipy.stats import norm
    if n1 == 0 or n2 == 0:
        return [float("nan"), float("nan")]
    p1, p2 = k1 / n1, k2 / n2
    p = (k1 + k2) / (n1 + n2)
    se = (p * (1 - p) * (1 / n1 + 1 / n2)) ** 0.5
    if se == 0:
        return [0.0, 1.0]
    z = (p1 - p2) / se
    return [float(z), float(2 * norm.sf(abs(z)))]


def mcnemar(b, c):
    """Exact McNemar on discordant pairs (b: ours fails only, c: crate fails only)."""
    from scipy.stats import binom
    n = b + c
    if n == 0:
        return 1.0
    return float(min(1.0, 2.0 * binom.cdf(min(b, c), n, 0.5)))


def phase_cpu(a):
    import stim
    from scipy.sparse import csr_matrix
    from relay_bp.bp import RelayDecoderF64
    prob = load_problem_dir(a.problem)
    H = prob.H.tocsc()
    m, n = H.shape
    k = prob.O.shape[0]
    priors = np.asarray(prob.priors, dtype=np.float64)
    sampler = stim.Circuit.from_file(os.path.join(a.problem, "circuit.stim")).compile_detector_sampler(
        seed=int(a.seed))
    det, obs = sampler.sample(a.shots, separate_observables=True, bit_packed=True)
    det = np.unpackbits(np.asarray(det, dtype=np.uint8), axis=1, count=m, bitorder="little").astype(np.uint8)
    obs = np.unpackbits(np.asarray(obs, dtype=np.uint8), axis=1, count=k, bitorder="little").astype(np.uint8)
    out = {"det": det, "obs": obs, "seed": np.array([a.seed], dtype=np.int64),
           "problem": np.array([os.path.basename(a.problem)])}
    Os = prob.O.tocsr()
    for s in [int(x) for x in a.s.split(",") if x]:
        dec = RelayDecoderF64(check_matrix=csr_matrix(H), error_priors=priors,
                              gamma0=a.gamma0, pre_iter=a.pre_iter, num_sets=a.legs,
                              set_max_iter=a.set_iter,
                              gamma_dist_interval=(a.gamma_lo, a.gamma_hi), stop_nconv=s)
        res = dec.decode_detailed_batch(det)
        corr = np.array([np.asarray(r.decoding, dtype=np.uint8) for r in res])
        cobs = np.asarray((Os @ corr.T).T % 2, dtype=np.uint8)
        out[f"crate_obs_s{s}"] = cobs
        out[f"crate_success_s{s}"] = np.array([bool(r.success) for r in res], dtype=np.uint8)
        out[f"crate_iters_s{s}"] = np.array([int(r.iterations) for r in res], dtype=np.int64)
        print(f"cpu phase: s={s} shots {a.shots} crate fails "
              f"{int(np.count_nonzero(np.any(cobs != obs, axis=1)))} "
              f"converged {int(out[f'crate_success_s{s}'].sum())} seed {a.seed}")
        sys.stdout.flush()
    np.savez_compressed(a.out, **out)
    print(f"cpu phase written {a.out}")
    return 0


def phase_gpu(a):
    import time
    from dbench.abi import AbiDecoder, AbiLibrary
    z = np.load(a.inp, allow_pickle=False)
    det = z["det"]
    obs = z["obs"]
    seed = int(z["seed"][0])
    prob = load_problem_dir(a.problem)
    print(f"# gpu phase: problem {os.path.basename(a.problem)} shots {det.shape[0]} stim seed {seed} "
          f"relay seed {a.relay_seed}")
    with open(a.out, "a", encoding="utf-8") as fo:
        for s in [int(x) for x in a.s.split(",") if x]:
            cfg = (f"decoder=bp;bp_method=ms;max_iter=1;relay={s};relay_legs={a.legs};"
                   f"relay_pre_iter={a.pre_iter};relay_set_iter={a.set_iter};"
                   f"relay_gamma0={a.gamma0};relay_gamma_lo={a.gamma_lo};relay_gamma_hi={a.gamma_hi};"
                   f"relay_seed={a.relay_seed}")
            dec = AbiDecoder(AbiLibrary(a.lib), cfg, prob)
            t0 = time.perf_counter()
            ours, _, st = dec.decode_batch(det)
            wall = time.perf_counter() - t0
            dec.close()
            ours_fail = np.any(ours != obs, axis=1)
            crate_obs = z[f"crate_obs_s{s}"]
            crate_fail = np.any(crate_obs != obs, axis=1)
            n = det.shape[0]
            of = int(np.count_nonzero(ours_fail))
            cf = int(np.count_nonzero(crate_fail))
            both = int(np.count_nonzero(ours_fail & crate_fail))
            only_ours = int(np.count_nonzero(ours_fail & ~crate_fail))
            only_crate = int(np.count_nonzero(~ours_fail & crate_fail))
            zt, zp = two_prop(of, n, cf, n)
            row = {"problem": os.path.basename(a.problem), "s": s, "shots": n,
                   "stim_seed": seed, "relay_seed": int(a.relay_seed), "config": cfg,
                   "ours": {"fail": of, "ler": of / n, "ci99": cp(of, n),
                            "converged": int(np.count_nonzero((st & 0xFFFF) == 0x0001)),
                            "iters_mean": float((st >> 16).mean()),
                            "wall_s": float(wall)},
                   "crate": {"fail": cf, "ler": cf / n, "ci99": cp(cf, n),
                             "converged": int(z[f"crate_success_s{s}"].sum()),
                             "iters_mean": float(z[f"crate_iters_s{s}"].mean())},
                   "discordant": {"both": both, "only_ours": only_ours, "only_crate": only_crate,
                                  "mcnemar_p": mcnemar(only_ours, only_crate)},
                   "two_proportion": {"z": zt, "p": zp}}
            print(json.dumps(row, sort_keys=True))
            fo.write(json.dumps(row, sort_keys=True) + "\n")
            fo.flush()
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("cpu", "gpu"))
    ap.add_argument("--problem", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shots", type=int, default=1000)
    ap.add_argument("--s", default="1,5")
    ap.add_argument("--lib", default="")
    ap.add_argument("--in", dest="inp", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--relay-seed", type=int, default=20261018)
    ap.add_argument("--gamma0", type=float, default=0.1)
    ap.add_argument("--pre-iter", type=int, default=80)
    ap.add_argument("--legs", type=int, default=300)
    ap.add_argument("--set-iter", type=int, default=60)
    ap.add_argument("--gamma-lo", type=float, default=-0.24)
    ap.add_argument("--gamma-hi", type=float, default=0.66)
    a = ap.parse_args(argv)
    if a.mode == "cpu":
        if not a.seed or not a.shots:
            raise SystemExit("cpu phase needs --seed and --shots")
        return phase_cpu(a)
    if not a.lib or not a.inp:
        raise SystemExit("gpu phase needs --lib and --in")
    return phase_gpu(a)


if __name__ == "__main__":
    sys.exit(main())
