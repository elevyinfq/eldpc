#!/usr/bin/env python3
"""One Relay-BP cell against BP + OSD-0 on one fresh stim shot set (a tool).

Decodes the SAME fresh shots with several decoders of the library and prints one JSON row per decoder:
shots, failures, LER, the 99 % Clopper-Pearson interval, the convergence count, the mean legs and
iterations per shot (legs for a relay handle: the test build's fastbp_test_relay_stats export), and
the single-shot latency distribution of a separate timed pass (one handle, back to back, W warm-up
shots then N timed shots; the host load average is printed).

Then one JSON comparison row per pair: the failure-count ratio, its 99 % interval (a ratio of two
independent binomial proportions: Clopper-Pearson lows/highs), the discordant counts and McNemar's
exact two-sided test on them.

Fresh shots: stim.Circuit.from_file(circuit.stim).compile_detector_sampler(seed).sample(n,
separate_observables=True, bit_packed=True); the seed is required and printed in every row.

usage: python3 -I t_relay_cells.py --lib <shipped .so> --problem <dir> --seed <64-bit> --shots N
        [--decoders relay1,bposd-a03125c3,bposd-a05ms31] [--stats-lib <relaydump .so>]
        [--relay-seed N] [--timing-shots N] [--tag NAME] [--out npz] [--jsonl FILE]
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
    return (lo, hi)


def dist(a):
    a = np.asarray(a, dtype=np.float64)
    if a.size == 0:
        return {}
    return {"n": int(a.size), "mean_us": float(a.mean()), "p50_us": float(np.percentile(a, 50)),
            "p99_us": float(np.percentile(a, 99)), "p999_us": float(np.percentile(a, 99.9)),
            "max_us": float(a.max()), "min_us": float(a.min())}


def loadavg():
    return [float(x) for x in open("/proc/loadavg").read().split()[:3]]


def config_of(name, args):
    if name.startswith("relay"):
        s = int(name[len("relay"):])
        return RELAY.format(s=s, legs=args.legs, pre=args.pre_iter, setit=args.set_iter,
                            gamma0=args.gamma0, lo=args.gamma_lo, hi=args.gamma_hi, seed=args.relay_seed)
    if name == "bposd-a03125c3":
        return BPOSD.format(mi=3, alpha=0.3125)
    if name == "bposd-a05ms31":
        return BPOSD.format(mi=31, alpha=0.5)
    raise SystemExit(f"unknown decoder {name}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib", required=True)
    ap.add_argument("--stats-lib", default="")
    ap.add_argument("--problem", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--shots", type=int, required=True)
    ap.add_argument("--decoders", default="relay1,bposd-a03125c3,bposd-a05ms31")
    ap.add_argument("--relay-seed", type=int, default=20261018)
    ap.add_argument("--gamma0", type=float, default=0.1)
    ap.add_argument("--pre-iter", type=int, default=80)
    ap.add_argument("--legs", type=int, default=300)
    ap.add_argument("--set-iter", type=int, default=60)
    ap.add_argument("--gamma-lo", type=float, default=-0.24)
    ap.add_argument("--gamma-hi", type=float, default=0.66)
    ap.add_argument("--timing-shots", type=int, default=0)
    ap.add_argument("--timing-seed", type=int, default=0,
                    help="a fresh stim seed for the timed pass (distinct shots, decoded once each); "
                         "0 = reuse the cell's shots")
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--jsonl", default="")
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
    hdr = {"tag": a.tag, "problem": os.path.basename(a.problem), "m": int(m), "n": int(n), "k": int(k),
           "nnz": int(H.nnz), "shots": int(a.shots), "stim_seed": int(a.seed),
           "relay_seed": int(a.relay_seed), "lib": a.lib, "stats_lib": a.stats_lib,
           "loadavg_start": loadavg()}
    print(json.dumps({"header": hdr}, sort_keys=True))
    sys.stdout.flush()

    fails = {}
    iters = {}
    legs = {}
    stats = {}
    rows = []
    for name in [x for x in a.decoders.split(",") if x]:
        cfg = config_of(name, a)
        dec = AbiDecoder(AbiLibrary(a.lib), cfg, prob)
        # test-build handle for the leg counters (relay only); outputs must agree shot for shot
        dec2 = None
        if name.startswith("relay") and a.stats_lib:
            dec2 = AbiDecoder(AbiLibrary(a.stats_lib), cfg, prob)
            lib2 = dec2.lib.lib
            import ctypes
            lib2.fastbp_test_relay_stats.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32),
                                                     ctypes.POINTER(ctypes.c_uint32),
                                                     ctypes.POINTER(ctypes.c_uint32)]
            lib2.fastbp_test_relay_stats.restype = ctypes.c_int
        f = np.zeros(a.shots, dtype=bool)
        it = np.zeros(a.shots, dtype=np.int64)
        cv = np.zeros(a.shots, dtype=bool)
        lg = np.zeros(a.shots, dtype=np.int64)
        diff = 0
        t0 = time.perf_counter()
        for s in range(a.shots):
            o, corr, st = dec.decode(det[s])
            f[s] = bool(np.any(o != obs[s]))
            it[s] = st >> 16
            cv[s] = (st & 0xFFFF) == 0x0001
            if dec2 is not None:
                o2, corr2, st2 = dec2.decode(det[s])
                c_found = ctypes.c_uint32(0)
                c_legs = ctypes.c_uint32(0)
                c_iters = ctypes.c_uint32(0)
                rc = lib2.fastbp_test_relay_stats(dec2._h, ctypes.byref(c_found), ctypes.byref(c_legs),
                                                  ctypes.byref(c_iters))
                if rc != 0:
                    raise SystemExit(f"fastbp_test_relay_stats rc {rc}")
                lg[s] = int(c_legs.value)
                if not (np.array_equal(o, o2) and np.array_equal(corr, corr2) and st == st2
                        and int(c_iters.value) == int(it[s])):
                    diff += 1
        wall = time.perf_counter() - t0
        nf = int(np.count_nonzero(f))
        row = {"kind": "decoder", "tag": a.tag, "problem": hdr["problem"], "decoder": name,
               "config": cfg, "shots": int(a.shots), "stim_seed": int(a.seed),
               "relay_seed": int(a.relay_seed), "fail": nf, "ler": nf / a.shots,
               "ci99": list(clopper_pearson(nf, a.shots)), "converged": int(np.count_nonzero(cv)),
               "iters_mean": float(it.mean()), "iters_p50": float(np.percentile(it, 50)),
               "batch_wall_s": float(wall),
               "stats_build_disagreements": int(diff)}
        if dec2 is not None:
            row["legs_mean"] = float(lg.mean())
            row["legs_p50"] = float(np.percentile(lg, 50))
            row["legs_max"] = int(lg.max())
        lat = None
        if a.timing_shots > 0:
            if a.timing_seed:
                tsampler = stim.Circuit.from_file(os.path.join(a.problem, "circuit.stim")) \
                    .compile_detector_sampler(seed=int(a.timing_seed))
                tdet, _ = tsampler.sample(a.timing_shots, separate_observables=True, bit_packed=True)
                tdet = np.unpackbits(np.asarray(tdet, dtype=np.uint8), axis=1, count=m,
                                     bitorder="little").astype(np.uint8)
                tshots = a.timing_shots
            else:
                tdet, tshots = det, a.shots
            dec3 = AbiDecoder(AbiLibrary(a.lib), cfg, prob)
            for i in range(min(a.warmup, tshots)):
                dec3.decode(tdet[i])
            v = []
            tstart = time.perf_counter()
            for i in range(a.timing_shots):
                q = time.perf_counter_ns()
                dec3.decode(tdet[i % tshots])
                v.append((time.perf_counter_ns() - q) / 1000.0)
            twall = time.perf_counter() - tstart
            lat = dist(v)
            lat["period_us"] = twall / a.timing_shots * 1e6
            lat["timing_seed"] = int(a.timing_seed or a.seed)
            lat["distinct_timed_shots"] = int(min(a.timing_shots, tshots))
            lat["loadavg_start"] = loadavg()
            lat["loadavg_end"] = loadavg()
            dec3.close()
        row["latency_us"] = lat
        rows.append(row)
        fails[name] = f
        iters[name] = it
        if dec2 is not None:
            legs[name] = lg
            dec2.close()
            stats[name] = "legs from " + a.stats_lib
        dec.close()
        print(json.dumps(row, sort_keys=True))
        sys.stdout.flush()

    # pair comparisons, the Relay-vs-BP+OSD-0 contrast first
    names = list(fails)
    comps = []
    from scipy.stats import binomtest
    base = "bposd-a03125c3" if "bposd-a03125c3" in fails else names[0]
    for other in names:
        if other == base:
            continue
        fb, fo = fails[base], fails[other]
        n01 = int(np.count_nonzero(fb & ~fo))    # base fails, other decodes
        n10 = int(np.count_nonzero(~fb & fo))    # other fails, base decodes
        nb, no = int(fb.sum()), int(fo.sum())
        cb, co = clopper_pearson(nb, a.shots), clopper_pearson(no, a.shots)
        c = {"kind": "comparison", "tag": a.tag, "problem": hdr["problem"], "shots": int(a.shots),
             "base": base, "other": other, "fail_base": nb, "fail_other": no,
             "ratio_base_over_other": (float(nb) / no) if no else None,
             "ratio_point": (float(nb) / no) if no else None,
             "ratio_ci99": [cb[0] / co[1], cb[1] / co[0]] if co[0] > 0 else None,
             "discordant_base_only": n01, "discordant_other_only": n10,
             "mcnemar_exact_p": float(binomtest(n01, n01 + n10, 0.5).pvalue) if (n01 + n10) else 1.0,
             "stim_seed": int(a.seed)}
        comps.append(c)
        print(json.dumps(c, sort_keys=True))
        sys.stdout.flush()

    if a.out:
        arrays = {"stim_seed": np.array([a.seed], dtype=np.uint64),
                  "shots": np.array([a.shots])}
        for nm in fails:
            arrays["fail_" + nm.replace("-", "_")] = fails[nm]
            arrays["iters_" + nm.replace("-", "_")] = iters[nm]
            if nm in legs:
                arrays["legs_" + nm.replace("-", "_")] = legs[nm]
        np.savez(a.out, **arrays)
    if a.jsonl:
        with open(a.jsonl, "a") as fh:
            for r in rows + comps:
                fh.write(json.dumps(r, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
