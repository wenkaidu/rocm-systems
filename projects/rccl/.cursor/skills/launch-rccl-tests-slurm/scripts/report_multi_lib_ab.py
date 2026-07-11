#!/usr/bin/env python3
"""A/B two multi-lib interleaved sweeps (each a results.csv from
run_rccl_tests_multi_lib.sh) to isolate the effect of one changed knob
(e.g. RCCL_GFX9_CHEAP_FENCE_OFF=0) per library.

For every (collective, algo/proto, lib) it computes the geomean over sizes of
the median out-of-place latency in each run, then reports:

  --mode delta   (default) delta% = (B - A)/A * 100 per lib.  Negative => B faster.
  --mode combined            each lib's delta% vs a reference lib, shown for A and
                             B side by side (A vs B tuning, both normalised to ref).

Usage:
  report_multi_lib_ab.py A_CSV B_CSV [--libs c1,c2,...] [--mode delta|combined]
                         [--ref COMMIT]
  (CSV args may be a results.csv file or the directory containing it.)
"""
import argparse, csv, math, os, statistics
from collections import defaultdict


def gmean(xs):
    xs = [x for x in xs if x and x > 0]
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float("nan")


def csv_path(p):
    return p if os.path.isfile(p) else os.path.join(p, "results.csv")


def load(path):
    tmp = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
    libs, colls = [], []
    for r in csv.DictReader(open(csv_path(path))):
        try:
            s = int(r["size_bytes"]); oop = float(r["oop_us"])
        except ValueError:
            continue
        if r["commit"] not in libs: libs.append(r["commit"])
        if r["collective"] not in colls: colls.append(r["collective"])
        pr = f'{r.get("algo","?")}/{r.get("proto","?")}'
        tmp[r["collective"]][pr][r["commit"]][s].append(oop)
    out = defaultdict(lambda: defaultdict(dict))
    for coll, pd in tmp.items():
        for pr, ld in pd.items():
            for lib, sd in ld.items():
                out[coll][pr][lib] = gmean([statistics.median(v) for v in sd.values()])
    return out, libs, colls


def dpct(x, r):
    return float("nan") if (math.isnan(x) or math.isnan(r) or r == 0) else (x - r) / r * 100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a"); ap.add_argument("b")
    ap.add_argument("--libs", default="")
    ap.add_argument("--mode", choices=["delta", "combined"], default="delta")
    ap.add_argument("--ref", default=None)
    args = ap.parse_args()

    A, alibs, colls = load(args.a)
    B, blibs, _ = load(args.b)
    libs = [l for l in (args.libs.split(",") if args.libs else alibs) if l in alibs and l in blibs]
    if not libs:
        print("no common libs between the two CSVs"); return
    colls = [c for c in colls if c in B]

    if args.mode == "delta":
        print("delta% = (B - A)/A * 100 per lib   [negative => B faster than A]")
        print(f"  A={csv_path(args.a)}\n  B={csv_path(args.b)}")
        print(f"  {'collective':>14} | {'algo/proto':>13} | " + " | ".join(f"{l:>11}" for l in libs))
        for coll in colls:
            for pr in sorted(set(A[coll]) & set(B[coll])):
                cells, ok = [], False
                for l in libs:
                    d = dpct(B[coll][pr].get(l, float("nan")), A[coll][pr].get(l, float("nan")))
                    cells.append(f"{d:>+11.1f}" if not math.isnan(d) else f"{'-':>11}"); ok |= not math.isnan(d)
                if ok:
                    print(f"  {coll.replace('_perf',''):>14} | {pr:>13} | " + " | ".join(cells))
        return

    ref = args.ref or libs[0]
    cands = [l for l in libs if l != ref]
    print(f"delta% vs ref={ref}; 'A' = first CSV tuning, 'B' = second CSV tuning")
    hdr = f"  {'collective':>14} | {'algo/proto':>13}"
    for c in cands: hdr += f" | {c+' A':>13} | {c+' B':>13}"
    print(hdr)
    for coll in colls:
        for pr in sorted(set(A[coll]) & set(B[coll])):
            ar, br = A[coll][pr].get(ref), B[coll][pr].get(ref)
            if not ar or not br: continue
            line, ok = f"  {coll.replace('_perf',''):>14} | {pr:>13}", False
            for c in cands:
                da = dpct(A[coll][pr].get(c, float("nan")), ar)
                db = dpct(B[coll][pr].get(c, float("nan")), br)
                line += " | " + (f"{da:>+13.1f}" if not math.isnan(da) else f"{'-':>13}")
                line += " | " + (f"{db:>+13.1f}" if not math.isnan(db) else f"{'-':>13}")
                ok |= not math.isnan(da)
            if ok: print(line)


if __name__ == "__main__":
    main()
