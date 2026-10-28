#!/usr/bin/env python3
"""The relay mode's determinism and statelessness check (a tool).

The ABI's rules the mode adds to: identically configured handles decode identically; the outputs
depend only on (problem, config, priors, syndrome) -- not on the handle's history, the call pattern
or the batch; dbench_decode_batch is a loop of dbench_decode; dbench_set_priors keeps the whole
handle consistent (a fresh handle created with p = 1/(1+exp(llr)) gives the same answer).

The shots are fresh (stim, the seed on the command line) and every comparison is bit for bit on
correction, observables and the status word.

usage: python -I t_relay_determinism.py --lib <so> --problem <dir> --seed <64-bit> [--shots N]
exit 0 iff every line agrees.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "python"))
from dbench.abi import AbiDecoder, AbiLibrary                 # noqa: E402
from dbench.formats import load_problem_dir                   # noqa: E402

CFG = ("decoder=bp;bp_method=ms;max_iter=1;relay=5;relay_legs=40;relay_pre_iter=20;"
       "relay_set_iter=10;relay_gamma0=0.1;relay_gamma_lo=-0.24;relay_gamma_hi=0.66;relay_seed=77")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib", required=True)
    ap.add_argument("--problem", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--shots", type=int, default=12)
    a = ap.parse_args(argv)
    import stim
    prob = load_problem_dir(a.problem)
    m = prob.H.shape[0]
    sampler = stim.Circuit.from_file(os.path.join(a.problem, "circuit.stim")).compile_detector_sampler(
        seed=int(a.seed))
    det, _ = sampler.sample(a.shots, separate_observables=True, bit_packed=True)
    det = np.unpackbits(np.asarray(det, dtype=np.uint8), axis=1, count=m, bitorder="little").astype(np.uint8)
    lib = AbiLibrary(a.lib)
    fails = 0

    # 1. two fresh handles, one shot at a time: identical
    h1 = AbiDecoder(lib, CFG, prob)
    h2 = AbiDecoder(lib, CFG, prob)
    # warm h2 with other shots first: history must not matter
    for i in range(min(4, a.shots)):
        h2.decode(det[(i + 3) % a.shots])
    ok = True
    for i in range(a.shots):
        o1, c1, s1 = h1.decode(det[i])
        o2, c2, s2 = h2.decode(det[i])
        if not (np.array_equal(o1, o2) and np.array_equal(c1, c2) and s1 == s2):
            ok = False
    print(f"two handles, history-independent: {'ok' if ok else 'MISMATCH'}")
    fails += 0 if ok else 1

    # 2. decode_batch == a loop of decode, on a third handle
    h3 = AbiDecoder(lib, CFG, prob)
    ob, cb, sb = h3.decode_batch(det)
    o4, c4, s4 = [], [], []
    h4 = AbiDecoder(lib, CFG, prob)
    for i in range(a.shots):
        o, c, s = h4.decode(det[i])
        o4.append(o)
        c4.append(c)
        s4.append(s)
    ok = (np.array_equal(ob, np.array(o4)) and np.array_equal(cb, np.array(c4))
          and np.array_equal(sb, np.array(s4, dtype=np.uint32)))
    print(f"decode_batch == loop of decode: {'ok' if ok else 'MISMATCH'}")
    fails += 0 if ok else 1

    # 3. set_priors: decode under llr = log((1-p)/p), then re-set the same priors on the handle
    llr = np.log((1.0 - np.asarray(prob.priors)) / np.asarray(prob.priors))
    h5 = AbiDecoder(lib, CFG, prob)
    ref = [h5.decode(det[i]) for i in range(a.shots)]
    h5.set_priors_llr(llr)
    ok = True
    for i in range(a.shots):
        o, c, s = h5.decode(det[i])
        if not (np.array_equal(o, ref[i][0]) and np.array_equal(c, ref[i][1]) and s == ref[i][2]):
            ok = False
    print(f"set_priors to the same priors: {'ok' if ok else 'MISMATCH'}")
    fails += 0 if ok else 1

    for h in (h1, h2, h3, h4, h5):
        h.close()
    print(f"relay determinism: {3 - fails}/3 lines ok")
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
