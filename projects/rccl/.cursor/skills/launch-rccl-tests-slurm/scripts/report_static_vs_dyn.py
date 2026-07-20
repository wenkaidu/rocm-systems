#!/usr/bin/env python3
"""Summarize a static-vs-dynamic rccl-tests comparison.

Reads results.csv (config,collective,cycle,order,size_bytes,oop_us,oop_busbw,
ip_us,ip_busbw,algo,proto,nchannels) and prints, per collective, a side-by-side
table of median out-of-place latency (us) for each config, with % delta vs the
baseline config (negative = faster than baseline). Also prints a geomean of
per-size ratios and a per-size algo/proto:#channels table (mode across cycles)
to confirm every config took the same path.

Usage: report_static_vs_dyn.py [results.csv] [baseline_tag]
  baseline_tag defaults to "static".
"""
import csv, sys, math
from collections import defaultdict

CSV = sys.argv[1] if len(sys.argv) > 1 else "results.csv"
BASELINE = sys.argv[2] if len(sys.argv) > 2 else "static"
METRIC = "oop_us"

def median(xs):
    xs = sorted(xs)
    n = len(xs)
    if n == 0: return None
    return xs[n//2] if n % 2 else 0.5*(xs[n//2-1]+xs[n//2])

def hsize(b):
    b = int(b); units = ["B","KB","MB","GB","TB"]; i = 0
    while b >= 1024 and i < len(units)-1:
        b /= 1024.0; i += 1
    return (f"{b:.0f}{units[i]}" if b == int(b) else f"{b:.1f}{units[i]}")

# data[coll][size][config] -> list of metric values
data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
# ap[coll][size][config] -> list of "ALGO/PROTO:nch" strings
ap = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
configs = []
with open(CSV) as f:
    for row in csv.DictReader(f):
        try:
            v = float(row[METRIC])
        except (ValueError, KeyError):
            continue
        coll = row["collective"]; size = int(row["size_bytes"]); cfg = row["config"]
        data[coll][size][cfg].append(v)
        if cfg not in configs: configs.append(cfg)
        a = row.get("algo"); p = row.get("proto"); n = row.get("nchannels")
        if a and a not in ("N/A", ""):
            ap[coll][size][cfg].append(f"{a}/{p}:{n}")

def mode(xs):
    if not xs: return "-"
    return max(set(xs), key=xs.count)

# Fall back to the first config if the requested baseline isn't present.
if BASELINE not in configs and configs:
    BASELINE = configs[0]
# Order configs: baseline first, then others as encountered.
order = ([BASELINE] if BASELINE in configs else []) + [c for c in configs if c != BASELINE]

for coll in sorted(data):
    print(f"\n===== {coll}  (median {METRIC}, us; delta vs {BASELINE}) =====")
    hdr = f"{'size':>9} " + " ".join(f"{c:>20}" for c in order)
    print(hdr)
    geo = defaultdict(list)  # cfg -> ratios vs baseline
    for size in sorted(data[coll]):
        meds = {c: median(data[coll][size].get(c, [])) for c in order}
        base = meds.get(BASELINE)
        cells = []
        for c in order:
            m = meds.get(c)
            if m is None:
                cells.append(f"{'-':>20}"); continue
            if c == BASELINE or not base:
                cells.append(f"{m:>10.2f}{'':>10}")
            else:
                d = 100.0*(m-base)/base
                geo[c].append(m/base)
                cells.append(f"{m:>10.2f}({d:+5.1f}%)")
        print(f"{hsize(size):>9} " + " ".join(cells))
    # geomean line
    print(f"{'GEOMEAN':>9} " + " ".join(
        (f"{'baseline':>20}" if c == BASELINE else
         (f"{(math.exp(sum(map(math.log,geo[c]))/len(geo[c]))-1)*100:>+19.1f}%" if geo.get(c) else f"{'-':>20}"))
        for c in order))

    # algo/proto table (mode across cycles): confirm which path each config took
    if any(ap[coll][s] for s in ap[coll]):
        print(f"\n----- {coll}  algo/proto:#channels (mode across cycles) -----")
        print(f"{'size':>9} " + " ".join(f"{c:>20}" for c in order))
        for size in sorted(data[coll]):
            cells = [f"{mode(ap[coll][size].get(c, [])):>20}" for c in order]
            print(f"{hsize(size):>9} " + " ".join(cells))
