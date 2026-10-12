#!/usr/bin/env python3
"""Aggregate reviewer-response runs: mean/std per config under both clustering
protocols, paired significance tests across seeds, and per-file stratification."""
import json, glob, os, itertools
import numpy as np

ART = os.path.join(os.path.dirname(os.path.abspath(__file__)), "core_artifacts")

def load():
    data = {}
    for fp in sorted(glob.glob(os.path.join(ART, "reviewer_response_seed*.json"))):
        seed = os.path.basename(fp).replace("reviewer_response_seed", "").replace(".json", "")
        data[seed] = json.load(open(fp))
    return data

def collect(data, cfg, protocol, metric):
    vals = []
    for seed, d in sorted(data.items()):
        if cfg in d and protocol in d[cfg]:
            vals.append(d[cfg][protocol][metric])
    return np.array(vals, dtype=float)

def paired_t(a, b):
    """Paired t-test; returns (t, p, cohen_dz). n is small (5 seeds) -- report honestly."""
    from scipy import stats
    if len(a) != len(b) or len(a) < 2:
        return None
    d = a - b
    if np.allclose(d, 0):
        return (0.0, 1.0, 0.0)
    t, p = stats.ttest_rel(a, b)
    dz = d.mean() / d.std(ddof=1) if d.std(ddof=1) > 0 else float("inf")
    return (float(t), float(p), float(dz))

def main():
    data = load()
    if not data:
        print("no results yet"); return
    seeds = sorted(data)
    cfgs = sorted({c for d in data.values() for c in d})
    print(f"seeds: {seeds}\n")

    for protocol in ["per_file_oracleK", "global_oracleK"]:
        print(f"===== {protocol} =====")
        print(f"{'config':32s} {'ARI mean±std':>18s} {'NMI mean±std':>18s}  n")
        for c in cfgs:
            a = collect(data, c, protocol, "ARI")
            n = collect(data, c, protocol, "NMI")
            if len(a) == 0:
                continue
            print(f"{c:32s} {a.mean():7.4f}±{a.std(ddof=1) if len(a)>1 else 0:.4f}   "
                  f"{n.mean():7.4f}±{n.std(ddof=1) if len(n)>1 else 0:.4f}  {len(a)}")
        print()

    # key paired comparisons (the reviewers' central questions)
    COMPARISONS = [
        ("simple_ALLedges",  "bert_ALLedges",  "R1.3/R2: handcrafted vs BERT (gold edges)"),
        ("simple_NOreply",   "bert_NOreply",   "R1.3/R2: handcrafted vs BERT (no gold edges)"),
        ("simple_ALLedges",  "simple_NOreply", "R1.1/R2.3: does removing gold reply edges hurt simple?"),
        ("bert_ALLedges",    "bert_NOreply",   "does removing gold reply edges hurt BERT?"),
        ("simple_ALLedges",  "constant_ALLedges", "do node features matter at all vs constant?"),
        ("combined_ALLedges","simple_ALLedges", "does adding BERT on top of handcrafted help?"),
    ]
    print("===== paired tests across seeds (per_file_oracleK, ARI) =====")
    print("NOTE: n=5 seeds. p-values are indicative, not confirmatory.\n")
    for x, y, label in COMPARISONS:
        a, b = collect(data, x, "per_file_oracleK", "ARI"), collect(data, y, "per_file_oracleK", "ARI")
        if len(a) < 2 or len(a) != len(b):
            print(f"  [pending] {label}"); continue
        r = paired_t(a, b)
        print(f"  {label}")
        print(f"    {x} = {a.mean():.4f}  vs  {y} = {b.mean():.4f}   "
              f"diff = {a.mean()-b.mean():+.4f}")
        print(f"    paired t={r[0]:+.3f}  p={r[1]:.4f}  Cohen dz={r[2]:+.2f}\n")

    # R3: stratify by speaker repetition
    print("===== R3: performance vs speaker repetition (per-file, simple_ALLedges) =====")
    rows = []
    for seed, d in data.items():
        for rec in d.get("simple_ALLedges", {}).get("per_file_detail", []):
            rows.append((rec["speaker_repetition"], rec["ARI"], rec["K"], rec["n"]))
    if rows:
        arr = np.array([[r[0], r[1]] for r in rows])
        med = np.median(arr[:, 0])
        lo, hi = arr[arr[:, 0] <= med, 1], arr[arr[:, 0] > med, 1]
        from scipy import stats
        rho, prho = stats.spearmanr(arr[:, 0], arr[:, 1])
        print(f"  files x seeds: {len(rows)}   median msgs/speaker = {med:.2f}")
        print(f"  low-repetition  files: mean ARI {lo.mean():.4f} (n={len(lo)})")
        print(f"  high-repetition files: mean ARI {hi.mean():.4f} (n={len(hi)})")
        print(f"  Spearman rho(speaker_repetition, ARI) = {rho:+.3f}  p={prho:.4f}")
    else:
        print("  [pending]")

if __name__ == "__main__":
    main()
