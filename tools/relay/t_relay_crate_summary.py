#!/usr/bin/env python3
"""Pool the crate-comparison rows and print the statistical check.

Reads the JSONL rows t_relay_crate2.py's gpu phase appends (one per problem and S, one per fresh
chunk), sums the failure and convergence counts over chunks, and prints, per (problem, S): the
pooled shots / failures / LER with its 99 % Clopper-Pearson interval for both decoders, the
two-proportion z-test on the pooled counts, the discordant-pair counts and the exact McNemar test.
Every number is printed here; nothing is computed by hand.

usage: python -I t_relay_crate_summary.py <results.jsonl>
"""
import json
import sys
from collections import defaultdict

import numpy as np
from scipy.stats import beta, binom, norm


def cp(k, n, alpha=0.01):
    if n <= 0:
        return [0.0, 1.0]
    lo = 0.0 if k == 0 else float(beta.ppf(alpha / 2.0, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1.0 - alpha / 2.0, k + 1, n - k))
    return [lo, hi]


def main(path):
    rows = [json.loads(line) for line in open(path) if line.strip().startswith("{")]
    agg = defaultdict(lambda: {"shots": 0, "of": 0, "cf": 0, "oc": 0, "cc": 0, "both": 0,
                               "only_ours": 0, "only_crate": 0, "oi": 0.0, "ci": 0.0})
    for r in rows:
        a = agg[(r["problem"], r["s"])]
        n = r["shots"]
        a["shots"] += n
        a["of"] += r["ours"]["fail"]
        a["cf"] += r["crate"]["fail"]
        a["both"] += r["discordant"]["both"]
        a["only_ours"] += r["discordant"]["only_ours"]
        a["only_crate"] += r["discordant"]["only_crate"]
        a["oi"] += r["ours"]["iters_mean"] * n
        a["ci"] += r["crate"]["iters_mean"] * n
    for (prob, s), a in sorted(agg.items()):
        n = a["shots"]
        oi = a["oi"] / n
        ci = a["ci"] / n
        p1, p2 = a["of"] / n, a["cf"] / n
        p = (a["of"] + a["cf"]) / (2 * n)
        se = (p * (1 - p) * (2.0 / n)) ** 0.5
        z = (p1 - p2) / se if se else 0.0
        pv = float(2 * norm.sf(abs(z))) if se else 1.0
        b, c = a["only_ours"], a["only_crate"]
        mcn = float(min(1.0, 2.0 * binom.cdf(min(b, c), b + c, 0.5))) if b + c else 1.0
        print(f"{prob} S={s}: shots {n}")
        print(f"  ours : fail {a['of']:4d} ler {p1:.6e} ci99 [{cp(a['of'], n)[0]:.6e}, {cp(a['of'], n)[1]:.6e}] "
              f"iters_mean {oi:.3f}")
        print(f"  crate: fail {a['cf']:4d} ler {p2:.6e} ci99 [{cp(a['cf'], n)[0]:.6e}, {cp(a['cf'], n)[1]:.6e}] "
              f"iters_mean {ci:.3f}")
        print(f"  two-proportion z {z:.4f} p {pv:.4g}; discordant both {a['both']} only_ours {b} "
              f"only_crate {c}; McNemar exact p {mcn:.4g}")
        print(f"  iteration-mean ratio ours/crate = {oi:.3f} / {ci:.3f} = {oi / ci:.4f}")


if __name__ == "__main__":
    main(sys.argv[1])
