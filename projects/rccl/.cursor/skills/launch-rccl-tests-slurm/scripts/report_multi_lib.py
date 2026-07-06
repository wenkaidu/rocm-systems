#!/usr/bin/env python3
"""Report a multi-lib interleaved rccl-tests sweep produced by
run_rccl_tests_multi_lib.sh.

For every collective it prints a per-size table of the median out-of-place
latency (us) for each library alongside the dominant algo/proto, then a
per-collective / per-protocol geomean summary with each candidate's delta vs a
reference library (so LL / LL128 / SIMPLE effects are isolated).

Usage:
    report_multi_lib.py [RESULTS_CSV] [--ref COMMIT] [--labels c1=name,c2=name]

Defaults: RESULTS_CSV = ~/rccl_mn_perf/multi_lib/results.csv
          --ref       = the first commit seen in the CSV (baseline column)
"""
import argparse, csv, math, os, statistics
from collections import Counter, defaultdict


def hz(b):
    b = int(b)
    if b >= 2**30 and b % 2**30 == 0: return f"{b//2**30}G"
    if b >= 2**20 and b % 2**20 == 0: return f"{b//2**20}M"
    if b >= 2**10 and b % 2**10 == 0: return f"{b//2**10}K"
    return str(b)


def gmean(xs):
    xs = [x for x in xs if x and x > 0]
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="?",
                    default=os.path.expanduser("~/rccl_mn_perf/multi_lib/results.csv"))
    ap.add_argument("--ref", default=None, help="baseline commit for delta%%")
    ap.add_argument("--labels", default="", help="comma list of commit=label")
    args = ap.parse_args()

    labels = {}
    for kv in filter(None, args.labels.split(",")):
        if "=" in kv:
            k, v = kv.split("=", 1); labels[k.strip()] = v.strip()

    rows = list(csv.DictReader(open(args.csv)))
    if not rows:
        print(f"no rows in {args.csv}"); return

    # preserve first-seen order of commits and collectives
    libs, colls = [], []
    for r in rows:
        if r["commit"] not in libs: libs.append(r["commit"])
        if r["collective"] not in colls: colls.append(r["collective"])
    ref = args.ref or libs[0]
    cands = [l for l in libs if l != ref]

    print(f"csv: {args.csv}")
    print(f"reference (baseline): {ref}  {labels.get(ref,'')}")
    for l in libs:
        print(f"  lib {l}  {labels.get(l,'')}")

    summ = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for coll in colls:
        lat, proto = {}, {}
        for r in rows:
            if r["collective"] != coll: continue
            try:
                s = int(r["size_bytes"]); oop = float(r["oop_us"])
            except ValueError:
                continue
            lat.setdefault(s, {}).setdefault(r["commit"], []).append(oop)
            proto.setdefault(s, Counter())[f"{r.get('algo','?')}/{r.get('proto','?')}"] += 1
        if not lat: continue
        ncyc = max((len(v) for s in lat for v in lat[s].values()), default=0)
        print(f"\n===== {coll.replace('_perf','')}  median oop us, n~{ncyc}/lib =====")
        print(f"  {'size':>6} | {'algo/proto':>13} | " + " | ".join(f"{l:>10}" for l in libs))
        for s in sorted(lat):
            pr = proto[s].most_common(1)[0][0] if proto.get(s) else "?"
            meds = {l: (statistics.median(lat[s][l]) if lat[s].get(l) else None) for l in libs}
            cells = " | ".join(f"{meds[l]:10.2f}" if meds[l] is not None else f"{'-':>10}" for l in libs)
            print(f"  {hz(s):>6} | {pr:>13} | {cells}")
            for l in libs:
                if meds.get(l) is not None:
                    summ[coll][pr][l].append(meds[l])

    if cands:
        print("\n\n===== per-collective / per-protocol geomean of median oop (us), delta% vs ref =====")
        print(f"  {'collective':>16} | {'algo/proto':>13} | {ref:>10} | "
              + " | ".join(f"{c:>10}" for c in cands))
        for coll in colls:
            for pr in sorted(summ[coll]):
                g_ref = gmean(summ[coll][pr][ref])
                if math.isnan(g_ref): continue
                cells = []
                for c in cands:
                    g = gmean(summ[coll][pr][c])
                    cells.append(f"{(g-g_ref)/g_ref*100:+10.1f}" if not math.isnan(g) else f"{'-':>10}")
                print(f"  {coll.replace('_perf',''):>16} | {pr:>13} | {g_ref:10.2f} | "
                      + " | ".join(cells))


if __name__ == "__main__":
    main()
