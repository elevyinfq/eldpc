#!/usr/bin/env python3
"""The derived numbers of the LER table (a tool).

Reads the JSONL rows t_relay_table.py prints and computes, per problem, the failure-count ratio of
the Relay rows to the BP+OSD-0 a03125c3 row (and of the two OSD rows to each other), plus the
99 % lower bound of a Relay cell with zero failures (a03125c3's lower CI edge over the Relay cell's
upper CI edge).  Every ratio is printed by this command; nothing is computed by hand.

usage: python -I t_relay_table_summary.py <table.jsonl>
"""
import json
import sys


def main(path):
    rows = [json.loads(line) for line in open(path) if line.strip().startswith("{")]
    by = {}
    for r in rows:
        by.setdefault(r["problem"], {})[r["decoder"]] = r
    for prob, d in by.items():
        ref = d.get("bposd-a03125c3")
        print(f"# {prob}")
        for name, r in d.items():
            if ref is None or name == "bposd-a03125c3":
                continue
            if r["fail"] > 0:
                print(f"  ratio a03125c3_failures / {name}_failures = {ref['fail']} / {r['fail']} = "
                      f"{ref['fail'] / r['fail']:.4f}")
            else:
                bound = ref["ci99"][0] / r["ci99"][1]
                print(f"  {name} has 0 failures: 99% lower bound of the ratio = a03125c3 CI low "
                      f"{ref['ci99'][0]:.6e} / {name} CI high {r['ci99'][1]:.6e} = {bound:.4f}")
        if "bposd-a05ms31" in d and ref is not None:
            print(f"  ratio a05ms31_failures / a03125c3_failures = {d['bposd-a05ms31']['fail']} / "
                  f"{ref['fail']} = {d['bposd-a05ms31']['fail'] / ref['fail']:.4f}")
            print(f"  ratio relay1_failures / relay5_failures = {d['relay1']['fail']} / "
                  f"{d['relay5']['fail']} = "
                  f"{(d['relay1']['fail'] / d['relay5']['fail']) if d['relay5']['fail'] else float('inf'):.4f}")
        print(f"  iter means: " + ", ".join(f"{n} {r['iters_mean']:.2f}" for n, r in d.items()))
        for name, r in d.items():
            lat = r.get("latency_us") or {}
            if lat:
                print(f"  lat {name}: mean {lat['mean_us']:.1f} p50 {lat['p50_us']:.1f} "
                      f"p99 {lat['p99_us']:.1f} p999 {lat['p999_us']:.1f} max {lat['max_us']:.1f} us")


if __name__ == "__main__":
    main(sys.argv[1])
