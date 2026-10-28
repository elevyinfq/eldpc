#!/usr/bin/env python3
"""STL / LDL SASS sites of each kernel, inside and outside the BP iteration loop (a tool).

The resource check asks for the STL / LDL site counts inside and outside the BP iteration loop; the
res_diff.py tool counts all sites of a kernel together.  This tool disassembles the library with
`cuobjdump -sass`, normalises the anonymous-namespace hash, and for every kernel finds its largest
backward-branch span -- the loop body that carries the BP iterations (the kernels' only loop of that
size; the column/check passes live inside it) -- and counts the STL / LDL instructions inside that
span and outside it.

    python3 -I t_relay_spill_sites.py <lib.so> [<lib2.so>]

The numbers are per kernel; a kernel with no backward branch reports loop=none and every site as
outside.
"""
from __future__ import annotations

import collections
import re
import subprocess
import sys

HASH = re.compile(r"GLOBAL__N__[0-9a-f]+_9_bp_gpu_cu_[0-9a-f]{8}")
FUNC = re.compile(r"^\s*Function : (\S+)")
INSN = re.compile(r"/\*([0-9a-f]{4,})\*/\s+(?:@!?U?P[T0-9]+\s+)?([A-Z][A-Z0-9_]*(?:\.[A-Z0-9_]+)*)")
TARGET = re.compile(r"\b0x([0-9a-f]{2,})\b")


def norm(n: str) -> str:
    return HASH.sub("GLOBAL__N__HASH_bp_gpu_cu", n)


def scan(path: str):
    r = subprocess.run(["cuobjdump", "-sass", path], capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"cuobjdump -sass {path} failed: {r.stderr[-500:]}")
    funcs = collections.OrderedDict()
    cur = None
    for line in r.stdout.splitlines():
        m = FUNC.match(line)
        if m:
            cur = norm(m.group(1))
            funcs[cur] = []
            continue
        m = INSN.search(line)
        if m and cur is not None:
            addr = int(m.group(1), 16)
            op = m.group(2).split(".")[0]
            funcs[cur].append((addr, op, line))
    return funcs


def sites(path: str):
    out = {}
    for fn, insns in scan(path).items():
        if not insns:
            continue
        # the largest backward-branch span: the BP iteration loop
        lo = hi = None
        for addr, op, line in insns:
            if op != "BRA":
                continue
            for t in TARGET.finditer(line):
                tgt = int(t.group(1), 16)
                if tgt < addr and (lo is None or addr - tgt > hi - lo):
                    lo, hi = tgt, addr
        stl = ldl = in_stl = in_ldl = 0
        for addr, op, line in insns:
            if op not in ("STL", "LDL"):
                continue
            inside = lo is not None and lo <= addr <= hi
            if op == "STL":
                stl += 1
                in_stl += inside
            else:
                ldl += 1
                in_ldl += inside
        out[fn] = {"stl": stl, "ldl": ldl, "stl_in": in_stl, "ldl_in": in_ldl,
                   "loop": (f"0x{lo:x}-0x{hi:x}" if lo is not None else "none")}
    return out


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        raise SystemExit(__doc__)
    data = [(p, sites(p)) for p in argv]
    names = sorted(set().union(*[set(d) for _, d in data]))
    for fn in names:
        line = f"{fn[:88]:88s}"
        for p, d in data:
            s = d.get(fn)
            if s is None:
                line += f" | {p}: absent"
            else:
                outside_stl = s["stl"] - s["stl_in"]
                outside_ldl = s["ldl"] - s["ldl_in"]
                line += (f" | STL {s['stl']} (in {s['stl_in']} / out {outside_stl})"
                         f" LDL {s['ldl']} (in {s['ldl_in']} / out {outside_ldl}) loop {s['loop']}")
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
