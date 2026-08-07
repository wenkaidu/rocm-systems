#!/usr/bin/env python3
"""Parse an env-var combination sweep (from run_env_combo_sweep.sh) into
per-node-count MEDIAN latency tables (Markdown).

Latency = the out-of-place `time` column (col 6) of rccl-tests output. For each
(node count, env combo, message size) the median across the CYCLES repeats is
reported (the median rejects outliers, per the usual "drop outliers" ask).

Usage:
    python3 parse_env_combo_latency.py <log_dir> [out.md]

<log_dir> is the sweep OUT dir containing att__*.log and sweep_meta.env.
If out.md is omitted it writes <log_dir>/latency_by_size.md.
"""
import os, re, sys, glob, collections, statistics

def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    logdir = os.path.abspath(sys.argv[1])
    dst = sys.argv[2] if len(sys.argv) > 2 else os.path.join(logdir, "latency_by_size.md")

    # var order (for column labels) from the manifest the runner wrote
    meta = {}
    mpath = os.path.join(logdir, "sweep_meta.env")
    if os.path.exists(mpath):
        for line in open(mpath):
            if "=" in line:
                k, _, v = line.strip().partition("=")
                meta[k] = v
    sweep_vars = meta.get("SWEEP_VARS", "").split()
    letters = [chr(ord("A") + i) for i in range(len(sweep_vars))]

    row_re = re.compile(r"^\s*(\d+)\s+\d+\s+\S+\s+\S+\s+\S+\s+([\d.]+)\s")

    # vals[N][size][combo_tuple] = [per-cycle time_us];  combo_tuple = (('VAR', val), ...)
    vals = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(list)))
    nodes, combos, sizes_by_n = set(), set(), collections.defaultdict(set)

    for fn in glob.glob(os.path.join(logdir, "att__*.log")):
        segs = os.path.basename(fn)[:-4].split("__")  # drop .log
        # segs = ['att', 'N<n>', 'VAR=VAL', ..., 'c<cycle>']
        N = int(segs[1][1:])
        combo = tuple((s.split("=", 1)[0], int(s.split("=", 1)[1])) for s in segs[2:-1])
        nodes.add(N); combos.add(combo)
        with open(fn) as fh:
            for line in fh:
                m = row_re.match(line)
                if not m:
                    continue
                size = int(m.group(1)); t = float(m.group(2))
                if size == 0:            # sub-rank-count sizes round to 0B
                    continue
                vals[N][size][combo].append(t)
                sizes_by_n[N].add(size)

    if not sweep_vars and combos:        # fall back to order seen in a filename
        sweep_vars = [k for k, _ in sorted(combos, key=len)[-1]]
        letters = [chr(ord("A") + i) for i in range(len(sweep_vars))]
    order = {v: i for i, v in enumerate(sweep_vars)}

    def norm(combo):  # sort a combo's (var,val) pairs by sweep-var order
        return tuple(sorted(combo, key=lambda kv: order.get(kv[0], 99)))

    def label(combo):
        c = dict(combo)
        return "".join(f"{letters[order[v]]}{c[v]}" for v in sweep_vars if v in c) or "base"

    combo_list = sorted(combos, key=lambda cb: tuple(v for _, v in norm(cb)))

    def human(sz):
        sz = int(sz)
        for u in ("B", "KB", "MB", "GB"):
            if sz < 1024 or u == "GB":
                return f"{sz}{u}" if u == "B" else f"{sz:g}{u}"
            sz /= 1024

    out = []
    out.append(f"# {meta.get('COLL','?')} latency sweep — median of "
               f"{meta.get('CYCLES','?')} cycles (out-of-place time, microseconds)\n")
    out.append(f"Lib: `{meta.get('LIB','?')}` | flags: `{meta.get('FLAGS','?')}` | "
               f"8 GPUs/node | per-size median. Lower = better.\n")
    out.append("Column legend: " +
               ", ".join(f"**{letters[i]}**=`{v}`" for i, v in enumerate(sweep_vars)) +
               " (digit after each letter = 0/1 setting).\n")

    for N in sorted(nodes):
        if not sizes_by_n[N]:
            continue
        out.append(f"\n## {N} node{'s' if N > 1 else ''} ({N*8} ranks)\n")
        cols = [label(cb) for cb in combo_list]
        out.append("| size | " + " | ".join(cols) + " |")
        out.append("|" + "---|" * (len(cols) + 1))
        for size in sorted(sizes_by_n[N]):
            cells = []
            for cb in combo_list:
                lst = vals[N][size].get(cb) or []
                cells.append(f"{statistics.median(lst):.2f}" if lst else "-")
            out.append(f"| {human(size)} | " + " | ".join(cells) + " |")

    txt = "\n".join(out) + "\n"
    with open(dst, "w") as fh:
        fh.write(txt)
    print(txt)
    print(f"\nSaved -> {dst}")


if __name__ == "__main__":
    main()
