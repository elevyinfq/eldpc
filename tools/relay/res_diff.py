#!/usr/bin/env python3
"""Per-kernel resource and spill-site diff of two builds.

A per-kernel resource diff that compares kernels by mangled name never matches one
(_GLOBAL__N__<hash> names differ between builds), so it would always report identical and
pass with a missing base.  This tool normalises the anonymous-namespace hash (and the
mangled length prefix), and FAILS when 0 kernels matched or when a file is missing.

    res_diff.py <base.so> <candidate.so> [--all]

For every kernel present in both builds it prints REG / STACK / LOCAL / SHARED (cuobjdump -res-usage)
and the count of STL / LDL SASS sites (a spill or a local-memory access shows here even when LOCAL is
0), and flags any growth: STACK, LOCAL or the STL+LDL site count may not grow for a fused
instantiation (the kernel's template argument 2 is SUM = 1), and REG may grow only up to the occupancy
cap (128 registers at 512 threads, one block per SM).  A kernel that exists in only one build is
reported too; --all lists every matched kernel, not only the ones that grew.
"""
import os
import re
import subprocess
import sys
from pathlib import Path

CUOBJDUMP = os.environ.get("CUOBJDUMP", "cuobjdump")
HASH = re.compile(r"GLOBAL__N__[0-9a-f]+_9_bp_gpu_cu_[0-9a-f]{8}")


def norm(name):
    """Normalise the per-build anonymous-namespace hash; keep the template arguments."""
    return HASH.sub("GLOBAL__N__HASH_bp_gpu_cu", name)


def res_usage(path):
    r = subprocess.run([CUOBJDUMP, "-res-usage", str(path)], capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"res_diff: cuobjdump -res-usage {path} failed:\n{r.stderr[-2000:]}")
    funcs, cur = {}, None
    for line in r.stdout.splitlines():
        m = re.match(r"\s*Function (\S+):", line)
        if m:
            cur = norm(m.group(1))
            funcs.setdefault(cur, {})
            continue
        if cur is not None:
            vals = {k: int(v) for k, v in re.findall(r"(REG|STACK|LOCAL|SHARED):(\d+)", line)}
            if vals:
                funcs[cur].update(vals)
    return funcs


def spill_sites(path):
    """Count STL (store-local) and LDL (load-local) SASS instructions per kernel."""
    r = subprocess.run([CUOBJDUMP, "-sass", str(path)], capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"res_diff: cuobjdump -sass {path} failed:\n{r.stderr[-2000:]}")
    out, cur = {}, None
    for line in r.stdout.splitlines():
        m = re.match(r"\s*Function : (\S+)", line)
        if m:
            cur = norm(m.group(1))
            out.setdefault(cur, 0)
            continue
        if cur is not None and re.search(r"\b(STL|LDL)\b", line):
            out[cur] += 1
    return out


def is_fused(name):
    """bp_kernel<RM, SUM, ILV, NT, SHSUM, RSLOT>: SUM = 1 is the fused formulation."""
    m = re.search(r"bp_kernelILi\d+ELi(\d+)E", name)
    return m is not None and m.group(1) == "1"


def main(argv):
    show_all = "--all" in argv
    argv = [a for a in argv if a != "--all"]
    if len(argv) != 2:
        print("usage: res_diff.py <base.so> <candidate.so> [--all]", file=sys.stderr)
        return 2
    base, cand = Path(argv[0]), Path(argv[1])
    for p in (base, cand):
        if not p.is_file():
            print(f"res_diff: FAIL: {p} missing", file=sys.stderr)
            return 2
    rb, rc = res_usage(base), res_usage(cand)
    sb, sc = spill_sites(base), spill_sites(cand)
    common = sorted(set(rb) & set(rc))
    if not common:
        print("res_diff: FAIL: 0 kernels matched by normalised name", file=sys.stderr)
        return 2
    only_b = sorted(set(rb) - set(rc))
    only_c = sorted(set(rc) - set(rb))
    print(f"res_diff: {base.name} vs {cand.name}: {len(common)} kernels matched, "
          f"{len(only_b)} only in base, {len(only_c)} only in candidate")
    for n in only_b:
        print(f"  only in base: {n}")
    for n in only_c:
        print(f"  only in candidate: {n}")
    fails, nonfused = [], []
    for n in common:
        b, c = rb[n], rc[n]
        bs, cs = sb.get(n, 0), sc.get(n, 0)
        grow = []
        for k in ("STACK", "LOCAL"):
            if c.get(k, 0) > b.get(k, 0):
                grow.append(f"{k} {b.get(k, 0)} -> {c.get(k, 0)}")
        if cs > bs:
            grow.append(f"STL/LDL sites {bs} -> {cs}")
        if is_fused(n) and c.get("REG", 0) > 128:
            grow.append(f"REG {b.get('REG', 0)} -> {c.get('REG', 0)} (over the 128 cap)")
        changed = (c.get("REG", 0) != b.get("REG", 0) or c.get("SHARED", 0) != b.get("SHARED", 0)
                   or grow)
        if grow:
            (fails if is_fused(n) else nonfused).append((n, grow))
        if show_all or changed:
            tag = "FUSED" if is_fused(n) else "     "
            print(f"  {tag} {n}")
            print(f"        base      REG {b.get('REG', 0):3d} STACK {b.get('STACK', 0):3d} "
                  f"LOCAL {b.get('LOCAL', 0):3d} SHARED {b.get('SHARED', 0):5d} STL/LDL {bs}")
            print(f"        candidate REG {c.get('REG', 0):3d} STACK {c.get('STACK', 0):3d} "
                  f"LOCAL {c.get('LOCAL', 0):3d} SHARED {c.get('SHARED', 0):5d} STL/LDL {cs}")
    if nonfused:
        print(f"res_diff: note: {len(nonfused)} non-fused kernel(s) grow (allowed by the resource "
              "rule; listed for the record):")
        for n, g in nonfused:
            print(f"  {n}: {'; '.join(g)}")
    if fails:
        print("res_diff: FAIL: growth in", len(fails), "fused kernel(s)")
        for n, g in fails:
            print(f"  {n}: {'; '.join(g)}")
        return 1
    print("res_diff: OK: no STACK / LOCAL / STL-LDL growth in any fused kernel; "
          "no fused kernel over the REG cap")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
