#!/usr/bin/env python3
"""
Experiments answering the ROCLING 2026 reviewer comments (submission 43).

Addresses, directly:
  R1.1 / R2.3  the missing control: simple-feature GNN WITHOUT gold reply edges
  R2 "matched" same GCN architecture, same graph, node features varied across
       {constant, handcrafted, BERT, combined}  -> isolates feature contribution
  R2 "edge ablations" reply-only / temporal-only / speaker-only  -> isolates edge contribution
  R1.3 normalization confound: all features standardized (train-fit) so the
       comparison is not confounded by feature scale; raw-BERT numbers from the
       submitted version are retained separately for continuity.

Writes core_artifacts/reviewer_response_seed{SEED}.json
Usage: REEVAL_SEED=42 python reviewer_response.py
"""
import os, sys, json, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import core_rerun as cr

RR_PATH = lambda: os.path.join(cr.ART, f"reviewer_response_seed{cr.SEED}.json")

def save_rr(name, obj):
    path = RR_PATH()
    results = json.load(open(path)) if os.path.exists(path) else {}
    results[name] = obj
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    pf = obj.get("per_file_oracleK", {})
    gl = obj.get("global_oracleK", {})
    print(f"[rr seed={cr.SEED}] {name}: per-file ARI={pf.get('ARI')} NMI={pf.get('NMI')} | "
          f"global ARI={gl.get('ARI')} NMI={gl.get('NMI')}", flush=True)

cr.save_result = save_rr   # run_gcn resolves this at call time


def build_graphs(rows, edge_types, ann_dir):
    """Per-file graphs with an arbitrary subset of {temporal, speaker, reply} edges.
    Structure only -- node features are attached later so one build serves many
    feature variants."""
    import torch
    from collections import defaultdict
    by_file = defaultdict(list)
    for i, r in enumerate(rows):
        by_file[r["file"]].append(i)
    graphs = []
    for fname, idxs in sorted(by_file.items()):
        idxs = sorted(idxs, key=lambda i: rows[i]["line"])
        n = len(idxs)
        line_pos = {rows[i]["line"]: k for k, i in enumerate(idxs)}
        edges = set()
        if "temporal" in edge_types:
            for k in range(n - 1):
                edges.add((k, k + 1))
        if "speaker" in edge_types:
            by_nick = defaultdict(list)
            for k, i in enumerate(idxs):
                by_nick[rows[i]["nick"]].append(k)
            for _, ks in by_nick.items():
                for a in range(len(ks)):
                    for b in range(a + 1, len(ks)):
                        edges.add((ks[a], ks[b]))
        if "reply" in edge_types:
            ann_path = os.path.join(ann_dir, f"{fname}.annotation.txt")
            if os.path.exists(ann_path):
                with open(ann_path) as f:
                    for row in f:
                        parts = row.split()
                        if len(parts) < 2:
                            continue
                        p, c = int(parts[0]), int(parts[1])
                        if p in line_pos and c in line_pos and p != c:
                            edges.add(tuple(sorted((line_pos[p], line_pos[c]))))
        src = [a for a, b in edges] + [b for a, b in edges] + list(range(n))
        dst = [b for a, b in edges] + [a for a, b in edges] + list(range(n))
        deg = np.zeros(n, dtype=np.float32)
        for s in src:
            deg[s] += 1.0
        norm = 1.0 / np.sqrt(deg)
        w = (norm[src] * norm[dst]).astype(np.float32)
        A = torch.sparse_coo_tensor(np.vstack([src, dst]), w, (n, n)).coalesce()
        slot, labels = {}, []
        for i in idxs:
            c = rows[i]["cluster"]
            if c not in slot:
                slot[c] = len(slot)
            labels.append(slot[c])
        nicks = [rows[i]["nick"] for i in idxs]
        spk_rep = len(nicks) / max(1, len(set(nicks)))  # msgs per unique speaker
        graphs.append({"file": fname, "idxs": idxs, "A": A,
                       "y": np.array(labels), "K": len(slot),
                       "spk_rep": spk_rep})
    return graphs


def make_features(kind, rows, idxs, emb):
    if kind == "constant":
        return np.ones((len(idxs), 1), dtype=np.float32)
    if kind == "simple":
        return cr.simple_features(rows, idxs)
    if kind == "bert":
        return emb[idxs].astype(np.float32)
    if kind == "combined":
        return np.hstack([cr.simple_features(rows, idxs),
                          emb[idxs].astype(np.float32)]).astype(np.float32)
    raise ValueError(kind)


def attach_features(kind, g_tr, g_te, rows_tr, rows_te, emb_tr, emb_te, standardize=True):
    """Attach node features; standardize with train-fit statistics (no test leakage)."""
    Xtr = [make_features(kind, rows_tr, g["idxs"], emb_tr) for g in g_tr]
    Xte = [make_features(kind, rows_te, g["idxs"], emb_te) for g in g_te]
    if standardize and kind != "constant":
        stacked = np.vstack(Xtr)
        mu, sd = stacked.mean(0), stacked.std(0)
        sd[sd < 1e-6] = 1.0
        Xtr = [(x - mu) / sd for x in Xtr]
        Xte = [(x - mu) / sd for x in Xte]
    for g, x in zip(g_tr, Xtr):
        g["X"] = x.astype(np.float32)
    for g, x in zip(g_te, Xte):
        g["X"] = x.astype(np.float32)
    return Xtr[0].shape[1]



def run_gcn_dual(graphs_train, graphs_eval, in_dim, max_slots, tag, epochs=50):
    """Train the same 2-layer GCN as core_rerun.run_gcn, but evaluate under BOTH
    clustering protocols so the comparison with embedding-only baselines is
    apples-to-apples (reviewer R1.2 / R2.4):
      per_file_oracleK : cluster each file into that file's own thread count
                         (the protocol the submitted paper used for GCN rows)
      global_oracleK   : cluster all test messages into the corpus-wide thread
                         count (the protocol the submitted paper used for the
                         K-Means / Triplet rows)
    Also records per-file ARI plus a speaker-repetition statistic so performance
    can be stratified by graph structure (reviewer R3)."""
    import torch
    import torch.nn as nn
    from sklearn.cluster import KMeans
    torch.manual_seed(cr.SEED)
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    class GCN(nn.Module):
        def __init__(self):
            super().__init__()
            self.w0 = nn.Linear(in_dim, 256)
            self.w1 = nn.Linear(256, max_slots)
            self.drop = nn.Dropout(0.3)
        def forward(self, A, X):
            h = torch.relu(torch.sparse.mm(A, self.w0(X)))
            h = self.drop(h)
            return torch.sparse.mm(A, self.w1(h)), h

    model = GCN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    lossf = nn.CrossEntropyLoss()
    for g in graphs_train + graphs_eval:
        g["At"] = g["A"].to(device)
        g["Xt"] = torch.tensor(g["X"], device=device)
        g["yt"] = torch.tensor(np.minimum(g["y"], max_slots - 1),
                               dtype=torch.long, device=device)
    for ep in range(epochs):
        model.train()
        total = 0.0
        for gi in np.random.permutation(len(graphs_train)):
            g = graphs_train[gi]
            opt.zero_grad()
            logits, _ = model(g["At"], g["Xt"])
            loss = lossf(logits, g["yt"])
            loss.backward(); opt.step()
            total += float(loss.detach())
        if ep % 25 == 0 or ep == epochs - 1:
            print(f"[{tag}] epoch {ep} loss {total/len(graphs_train):.4f}", flush=True)

    model.eval()
    all_true, pred_perfile, all_emb, per_file = [], [], [], []
    correct = count = 0
    with torch.no_grad():
        for g in graphs_eval:
            logits, hidden = model(g["At"], g["Xt"])
            am = logits.argmax(1).cpu().numpy()
            correct += int((am == np.minimum(g["y"], max_slots - 1)).sum())
            count += len(am)
            H = hidden.cpu().numpy()
            all_emb.append(H)
            km = KMeans(n_clusters=g["K"], random_state=cr.SEED, n_init=10).fit(H)
            all_true += [f'{g["file"]}:{c}' for c in g["y"]]
            pred_perfile += [f'{g["file"]}:{c}' for c in km.labels_]
            per_file.append({"file": g["file"], "n": len(g["y"]), "K": int(g["K"]),
                             "speaker_repetition": float(g.get("spk_rep", float("nan"))),
                             "ARI": cr.cluster_metrics(list(g["y"]), list(km.labels_))["ARI"]})

    X = np.vstack(all_emb)
    # global protocol: one clustering over the whole test split
    gk = len(set(all_true))
    km_g = KMeans(n_clusters=gk, random_state=cr.SEED, n_init=10).fit(X)
    res = {
        "node_classification_accuracy": round(correct / count, 4),
        "per_file_oracleK": cr.cluster_metrics(all_true, pred_perfile, X),
        "global_oracleK": cr.cluster_metrics(all_true, list(km_g.labels_), X),
        "global_K_used": gk,
        "per_file_detail": per_file,
    }
    for g in graphs_train + graphs_eval:
        del g["At"], g["Xt"], g["yt"]
    save_rr(tag, res)



def baselines_both_protocols(rows_te, emb_te):
    """K-Means on frozen features under BOTH clustering protocols, so the
    embedding-only rows are directly comparable to the GCN rows (R1.2 / R2.4).
    The submitted paper reported these under the global protocol only, while
    GCN rows used the per-file protocol -- that mismatch is what we correct."""
    from sklearn.cluster import KMeans
    from sklearn.feature_extraction.text import TfidfVectorizer
    from collections import defaultdict

    y = [r["cluster"] for r in rows_te]
    by_file = defaultdict(list)
    for i, r in enumerate(rows_te):
        by_file[r["file"]].append(i)

    # keep TF-IDF sparse: k=961 K-Means over 20k dense dims is prohibitively slow
    tfidf = TfidfVectorizer(max_features=20000).fit_transform(
        [r["text"] for r in rows_te])
    feats = {"bertlarge": emb_te.astype(np.float32), "tfidf": tfidf}

    for name, F in feats.items():
        gk = len(set(y))
        km = KMeans(n_clusters=gk, random_state=cr.SEED, n_init=10).fit(F)
        glob = cr.cluster_metrics(y, list(km.labels_), None)

        true_pf, pred_pf = [], []
        for fname, idxs in sorted(by_file.items()):
            sub_true = [rows_te[i]["cluster"] for i in idxs]
            k = len(set(sub_true))
            lab = KMeans(n_clusters=k, random_state=cr.SEED,
                         n_init=10).fit_predict(F[idxs])
            true_pf += sub_true
            pred_pf += [f"{fname}:{c}" for c in lab]
        perfile = cr.cluster_metrics(true_pf, pred_pf, None)

        save_rr(f"kmeans_{name}", {"per_file_oracleK": perfile,
                                   "global_oracleK": glob,
                                   "global_K_used": gk})


def main():
    rows_tr, rows_te = cr.load_split("train"), cr.load_split("test")
    emb_tr = np.load(os.path.join(cr.ART, "emb_large_train.npy"))
    emb_te = np.load(os.path.join(cr.ART, "emb_large_test.npy"))
    tr_dir = os.path.join(cr.DATA, "train")
    te_dir = os.path.join(cr.DATA, "test")

    EDGE_SETS = {
        "ALLedges":     {"temporal", "speaker", "reply"},
        "NOreply":      {"temporal", "speaker"},
        "replyonly":    {"reply"},
        "temporalonly": {"temporal"},
        "speakeronly":  {"speaker"},
    }
    # (edge_set, [feature kinds]) -- matched design + edge ablations
    PLAN = [
        ("ALLedges", ["constant", "simple", "bert", "combined"]),
        ("NOreply",  ["constant", "simple", "bert", "combined"]),
        ("replyonly",    ["simple"]),
        ("temporalonly", ["simple"]),
        ("speakeronly",  ["simple"]),
    ]

    done = set()
    if os.path.exists(RR_PATH()):
        done = set(json.load(open(RR_PATH())).keys())

    if "kmeans_bertlarge" not in done:
        baselines_both_protocols(rows_te, emb_te)

    # max_slots must match the submitted paper's setting for comparability
    g_probe = build_graphs(rows_tr, EDGE_SETS["ALLedges"], tr_dir)
    max_slots = min(200, max(g["K"] for g in g_probe))
    print(f"[rr] max_slots={max_slots}", flush=True)
    del g_probe

    for edge_name, kinds in PLAN:
        pending = [k for k in kinds if f"{k}_{edge_name}" not in done]
        if not pending:
            print(f"[rr] {edge_name}: all done, skipping", flush=True)
            continue
        t0 = time.time()
        g_tr = build_graphs(rows_tr, EDGE_SETS[edge_name], tr_dir)
        g_te = build_graphs(rows_te, EDGE_SETS[edge_name], te_dir)
        print(f"[rr] built graphs {edge_name} in {time.time()-t0:.0f}s", flush=True)
        for kind in pending:
            tag = f"{kind}_{edge_name}"
            dim = attach_features(kind, g_tr, g_te, rows_tr, rows_te, emb_tr, emb_te)
            np.random.seed(cr.SEED)
            t1 = time.time()
            run_gcn_dual(g_tr, g_te, dim, max_slots, tag)
            print(f"[rr] {tag} ({dim}-dim) done in {time.time()-t1:.0f}s", flush=True)
        del g_tr, g_te


if __name__ == "__main__":
    main()
