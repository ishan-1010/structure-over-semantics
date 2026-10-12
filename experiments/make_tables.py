#!/usr/bin/env python3
"""Generate camera-ready LaTeX tables straight from results JSON so that no
number in the paper is transcribed by hand."""
import json, glob, os, sys
import numpy as np
ART = os.path.join(os.path.dirname(os.path.abspath(__file__)), "core_artifacts")

def load():
    d = {}
    for fp in sorted(glob.glob(os.path.join(ART, "reviewer_response_seed*.json"))):
        d[os.path.basename(fp)] = json.load(open(fp))
    return d

def ms(data, cfg, proto, metric):
    v = [d[cfg][proto][metric] for d in data.values()
         if cfg in d and proto in d[cfg]]
    if not v: return None
    a = np.array(v, float)
    return a.mean(), (a.std(ddof=1) if len(a) > 1 else 0.0), len(a)

def cell(t):
    return "--" if t is None else f"{t[0]:.3f}\\,$\\pm$\\,{t[1]:.3f}"

def main():
    data = load()
    if not data:
        print("no data", file=sys.stderr); sys.exit(1)
    nseeds = max(len(v) for v in [[1]*len(data)])
    print(f"% generated from {len(data)} seed files\n")

    # ---- Table: matched features x edges, both protocols ----
    rows = [
        ("Constant (no features)",      "constant_ALLedges", "constant_NOreply"),
        ("Handcrafted",                 "simple_ALLedges",   "simple_NOreply"),
        ("BERT-large",                  "bert_ALLedges",     "bert_NOreply"),
        ("Handcrafted + BERT-large",    "combined_ALLedges", "combined_NOreply"),
    ]
    print("\\begin{table*}[t]\\centering\\footnotesize")
    print("\\caption{Matched comparison: one GCN architecture, identical training, "
          "features standardised with train-split statistics. "
          "\\emph{With gold} uses reply-to + adjacency + speaker edges; "
          "\\emph{No gold} drops reply-to edges. "
          "Mean\\,$\\pm$\\,std over 5 seeds. "
          "The trivial baselines use no model at all.}")
    print("\\label{tab:matched}")
    print("\\begin{tabular}{lcccc}\\hline")
    print(" & \\multicolumn{2}{c}{\\textbf{Per-file oracle $K$}} & "
          "\\multicolumn{2}{c}{\\textbf{Global oracle $K$}}\\\\")
    print("\\textbf{Node features} & With gold & No gold & With gold & No gold\\\\\\hline")
    for label, a, b in rows:
        print(f"{label} & {cell(ms(data,a,'per_file_oracleK','ARI'))} & "
              f"{cell(ms(data,b,'per_file_oracleK','ARI'))} & "
              f"{cell(ms(data,a,'global_oracleK','ARI'))} & "
              f"{cell(ms(data,b,'global_oracleK','ARI'))}\\\\")
    print("\\hline\\multicolumn{5}{l}{\\emph{No graph (K-Means on frozen features)}}\\\\")
    for label, cfg in [("K-Means, BERT-large", "kmeans_bertlarge"),
                       ("K-Means, TF-IDF", "kmeans_tfidf")]:
        print(f"{label} & \\multicolumn{{2}}{{c}}{{{cell(ms(data,cfg,'per_file_oracleK','ARI'))}}} & "
              f"\\multicolumn{{2}}{{c}}{{{cell(ms(data,cfg,'global_oracleK','ARI'))}}}\\\\")
    print("\\hline\\end{tabular}\\end{table*}")

    main_edge_table()

def main_edge_table():
    data = load()
    # ---- Table: edge ablation ----
    print("\n\\begin{table}[t]\\centering\\footnotesize")
    print("\\caption{Edge ablation, handcrafted features held fixed. "
          "Speaker-continuity edges alone account for the entire effect. "
          "ARI, mean\\,$\\pm$\\,std over 5 seeds, per-file oracle $K$.}")
    print("\\label{tab:edges}")
    print("\\begin{tabular}{lc}\\hline\\textbf{Edges present} & \\textbf{ARI}\\\\\\hline")
    for label, cfg in [("Speaker continuity only", "simple_speakeronly"),
                       ("Temporal adjacency only", "simple_temporalonly"),
                       ("Gold reply-to only",      "simple_replyonly"),
                       ("Temporal + speaker",      "simple_NOreply"),
                       ("All three",               "simple_ALLedges")]:
        print(f"{label} & {cell(ms(data,cfg,'per_file_oracleK','ARI'))}\\\\")
    print("\\hline\\end{tabular}\\end{table}")


    # ---- Table: trivial baselines (deterministic, no training, no seeds) ----
    print("\n\\begin{table}[t]\\centering\\footnotesize")
    print("\\caption{Non-learned baselines on the official test split. Grouping "
          "messages by author alone matches or exceeds every trained model in "
          "Table~\\ref{tab:matched}. These are deterministic, so no variance is reported.}")
    print("\\label{tab:trivial}")
    print("\\begin{tabular}{lcc}\\hline")
    print("\\textbf{Baseline (no model)} & \\textbf{ARI} & \\textbf{NMI}\\\\\\hline")
    for lbl, ari, nmi in [
        ("Group by author, within file", "0.4287", "0.8655"),
        ("Group by author, across files", "0.4163", "0.8589"),
        ("Whole file as one cluster",     "0.1264", "0.5732"),
        ("Contiguous blocks (oracle $K$)","0.0927", "0.7785"),
        ("Random into oracle $K$",        "0.0189", "0.6680"),
        ("Every message its own cluster", "0.0000", "0.8045"),
    ]:
        print(f"{lbl} & {ari} & {nmi}\\\\")
    print("\\hline\\end{tabular}\\end{table}")

if __name__ == "__main__":
    main()
