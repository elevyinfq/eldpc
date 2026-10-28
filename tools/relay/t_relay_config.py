#!/usr/bin/env python3
"""The relay config key's parse and refusal rules (a tool).

The ABI's rule for a new key: it is parsed like the reference decoder's keys, an unknown value is rejected, and a
key that does not apply is refused.  This tool creates handles on a small dev problem with a list
of valid and invalid configs and prints one line per case; exit 0 iff every expectation holds.

    python3 -I t_relay_config.py --lib <core .so> --problem <problem dir, e.g. a small bb72 z problem>
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "python"))
from dbench.abi import AbiDecoder, AbiLibrary, AbiError   # noqa: E402
from dbench.formats import load_problem_dir               # noqa: E402

BASE = "decoder=bp;bp_method=ms;max_iter=1"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib", required=True)
    ap.add_argument("--problem", required=True)
    a = ap.parse_args(argv)
    prob = load_problem_dir(a.problem)
    lib = AbiLibrary(a.lib)
    cases = [
        # (config, expect_ok, what)
        (BASE, True, "no relay key at all (plain BP)"),
        (BASE + ";relay=0", True, "relay=0 (the mode off)"),
        (BASE + ";relay=1", True, "relay=1"),
        (BASE + ";relay=5;relay_legs=10;relay_pre_iter=2;relay_set_iter=2;relay_seed=7", True,
         "relay=5 with every companion key"),
        (BASE + ";relay=65535", True, "relay=65535 (the top of the range)"),
        (BASE + ";relay=abc", False, "relay=abc (not an integer)"),
        (BASE + ";relay=-1", False, "relay=-1 (below the range)"),
        (BASE + ";relay=65536", False, "relay=65536 (above the range)"),
        (BASE + ";relay=1.5", False, "relay=1.5 (not an integer)"),
        (BASE + ";relay=0;relay_legs=4", False, "a relay_* key without relay > 0"),
        (BASE + ";relay=1;relay_legs=0", False, "relay_legs below 1"),
        (BASE + ";relay=1;relay_pre_iter=0", False, "relay_pre_iter below 1"),
        (BASE + ";relay=1;relay_set_iter=0", False, "relay_set_iter below 1"),
        (BASE + ";relay=1;relay_gamma0=nan", False, "relay_gamma0=nan"),
        (BASE + ";relay=1;relay_gamma_lo=0.7;relay_gamma_hi=0.2", False, "gamma_lo > gamma_hi"),
        (BASE + ";relay=1;relay_seed=-3", False, "relay_seed below 0"),
        ("decoder=bposd;bp_method=ms;max_iter=1;osd_method=osd0;osd_order=0;relay=1", False,
         "relay > 0 on decoder=bposd"),
        (BASE + ";relay=1;bogus_key=2", False, "an unknown key beside relay"),
    ]
    bad = 0
    for cfg, want_ok, what in cases:
        got_ok, detail = True, ""
        try:
            d = AbiDecoder(lib, cfg, prob)
            d.close()
        except AbiError as e:
            got_ok, detail = False, str(e).splitlines()[0][:160]
        ok = got_ok == want_ok
        bad += 0 if ok else 1
        print(f"{'ok  ' if ok else 'FAIL'} expect {'accept' if want_ok else 'refuse'}: {what}"
              + (f"  [{detail}]" if detail else ""))
    print(f"relay config rules: {len(cases) - bad}/{len(cases)} as expected")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
