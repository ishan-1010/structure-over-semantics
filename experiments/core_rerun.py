#!/usr/bin/env python3
"""
Faithful rerun of the paper's core pipeline on the real Kummerfeld
irc-disentanglement corpus.

Implements, per the paper (conference_101719.tex):
  - K-Means baseline on BERT CLS features (Sec 4.1 / 5.2)
  - Triplet metric learning, BERT-base 768-dim init (Sec 4.3 / 5.4)
  - Siamese pairwise link prediction, BERT-large 1024-dim (Sec 4.2 / 5.3)
  - GNN with simple features vs hybrid GCN + BERT-large CLS (Sec 4.4-4.5 / 5.5)
    2-layer GCN, symmetric normalization, ReLU, dropout 0.3, Adam lr 1e-3,
    50 epochs; edges = reply-to (gold), temporal adjacency, speaker continuity.

Where the paper is ambiguous, both variants are computed and reported:
  - test graph WITH gold reply edges (paper's literal design; leaks labels)
    vs WITHOUT (adjacency + speaker only)
  - clustering with oracle K vs K estimated from classifier argmax

Artifacts land in experiments/core_artifacts/. Every phase checkpoints.
Usage: core_rerun.py [parse|embed|baseline|gcn|siamese|triplet|report|all]
"""
import json
import os
import re
import sys
import time
from collections import defaultdict

import numpy as np

SEED = int(os.environ.get("REEVAL_SEED", "42"))
ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "..", "irc_data", "data")
ART = os.path.join(ROOT, "core_artifacts")
os.makedirs(ART, exist_ok=True)

np.random.seed(SEED)

NICK_RE = re.compile(r"^\[\d\d:\d\d\] <([^>]+)> ?(.*)$")
ACTION_RE = re.compile(r"^\[\d\d:\d\d\] \* (\S+) ?(.*)$")
SYSTEM_RE = re.compile(r"^=== (\S+) ?(.*)$")


def parse_line(raw):
    m = NICK_RE.match(raw)
    if m:
        return m.group(1), m.group(2)
    m = ACTION_RE.match(raw)
    if m:
        return m.group(1), m.group(2)
    m = SYSTEM_RE.match(raw)
    if m:
        return m.group(1), raw[4:]
    return "unknown", raw


class UnionFind:
    def __init__(self):
        self.p = {}

    def find(self, x):
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[ra] = rb


def load_gold_clusters(path):
    """gold.*.clusters.txt -> {file: {line: cluster_id}}"""
    out = defaultdict(dict)
    with open(path) as f:
        for cid, row in enumerate(f):
            row = row.strip()
            if not row:
                continue
            fname, lines = row.split(":", 1)
            for ln in lines.split():
                out[fname][int(ln)] = cid
    return out


def train_clusters_from_annotations(ann_path):
    """Union-find over reply links; returns {line: cluster_root} for lines >= 1000."""
    uf = UnionFind()
    children = set()
    with open(ann_path) as f:
        for row in f:
            parts = row.split()
            if len(parts) < 2:
                continue
            parent, child = int(parts[0]), int(parts[1])
            uf.union(parent, child)
            children.add(child)
    return {ln: uf.find(ln) for ln in children if ln >= 1000}


def phase_parse():
    splits = {}
    # dev/test: gold cluster files are the canonical labels
    for split in ("dev", "test"):
        gold = load_gold_clusters(os.path.join(DATA, f"gold.{split}.clusters.txt"))
        rows = []
        for fname, line_map in sorted(gold.items()):
            ascii_path = os.path.join(DATA, split, f"{fname}.ascii.txt")
            with open(ascii_path, errors="replace") as f:
                raw_lines = f.read().split("\n")
            for ln, cid in sorted(line_map.items()):
                nick, text = parse_line(raw_lines[ln])
                rows.append({"file": fname, "line": ln, "nick": nick,
                             "text": text[:512], "cluster": f"{fname}:{cid}"})
        splits[split] = rows
    # train: derive clusters from annotation files
    rows = []
    train_dir = os.path.join(DATA, "train")
    for ann in sorted(os.listdir(train_dir)):
        if not ann.endswith(".annotation.txt"):
            continue
        fname = ann[: -len(".annotation.txt")]
        cluster_map = train_clusters_from_annotations(os.path.join(train_dir, ann))
        with open(os.path.join(train_dir, f"{fname}.ascii.txt"), errors="replace") as f:
            raw_lines = f.read().split("\n")
        for ln, root in sorted(cluster_map.items()):
            if ln >= len(raw_lines):
                continue
            nick, text = parse_line(raw_lines[ln])
            rows.append({"file": fname, "line": ln, "nick": nick,
                         "text": text[:512], "cluster": f"{fname}:{root}"})
    splits["train"] = rows

    for split, rows in splits.items():
        with open(os.path.join(ART, f"{split}.jsonl"), "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        n_clusters = len({r["cluster"] for r in rows})
        n_files = len({r["file"] for r in rows})
        print(f"[parse] {split}: {len(rows)} messages, {n_clusters} threads, "
              f"{n_files} files")


def load_split(split):
    rows = []
    with open(os.path.join(ART, f"{split}.jsonl")) as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def phase_embed():
    import torch
    from transformers import AutoModel, AutoTokenizer

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    for model_name, dim, tag in (("bert-large-uncased", 1024, "large"),
                                 ("bert-base-uncased", 768, "base")):
        tok = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name).to(device).eval()
        for split in ("test", "dev", "train"):
            out_path = os.path.join(ART, f"emb_{tag}_{split}.npy")
            if os.path.exists(out_path):
                print(f"[embed] {tag}/{split} exists, skipping")
                continue
            rows = load_split(split)
            texts = [r["text"] if r["text"].strip() else "empty" for r in rows]
            embs = np.zeros((len(texts), dim), dtype=np.float32)
            t0 = time.time()
            with torch.no_grad():
                for i in range(0, len(texts), 64):
                    batch = tok(texts[i:i + 64], padding=True, truncation=True,
                                max_length=64, return_tensors="pt").to(device)
                    out = model(**batch).last_hidden_state[:, 0, :]  # [CLS]
                    embs[i:i + 64] = out.float().cpu().numpy()
                    if i % 6400 == 0:
                        print(f"[embed] {tag}/{split} {i}/{len(texts)} "
                              f"({time.time()-t0:.0f}s)", flush=True)
            np.save(out_path, embs)
            print(f"[embed] {tag}/{split} done: {embs.shape} "
                  f"in {time.time()-t0:.0f}s")
        del model
        if device == "mps":
            torch.mps.empty_cache()


def cluster_metrics(labels_true, labels_pred, X=None):
    from sklearn.metrics import (adjusted_rand_score, homogeneity_completeness_v_measure,
                                 normalized_mutual_info_score, silhouette_score)
    ari = adjusted_rand_score(labels_true, labels_pred)
    nmi = normalized_mutual_info_score(labels_true, labels_pred)
    h, c, v = homogeneity_completeness_v_measure(labels_true, labels_pred)
    sil = None
    if X is not None and len(set(labels_pred)) > 1:
        try:
            sil = float(silhouette_score(X, labels_pred,
                                         sample_size=min(5000, len(X)),
                                         random_state=SEED))
        except Exception:
            sil = None
    return {"ARI": round(ari, 4), "NMI": round(nmi, 4), "Homogeneity": round(h, 4),
            "Completeness": round(c, 4), "V": round(v, 4),
            "Silhouette": None if sil is None else round(sil, 4)}


def save_result(name, obj):
    path = os.path.join(ART, f"results_seed{SEED}.json")
    results = {}
    if os.path.exists(path):
        with open(path) as f:
            results = json.load(f)
    results[name] = obj
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[result seed={SEED}] {name}: {json.dumps(obj)}")


def phase_baseline():
    from sklearn.cluster import KMeans
    from sklearn.feature_extraction.text import TfidfVectorizer

    rows = load_split("test")
    y = [r["cluster"] for r in rows]
    K = len(set(y))
    X = np.load(os.path.join(ART, "emb_large_test.npy"))
    km = KMeans(n_clusters=K, random_state=SEED, n_init=10).fit(X)
    save_result("kmeans_bert_large_oracleK", cluster_metrics(y, km.labels_, X))

    Xb = np.load(os.path.join(ART, "emb_base_test.npy"))
    km = KMeans(n_clusters=K, random_state=SEED, n_init=10).fit(Xb)
    save_result("kmeans_bert_base_oracleK", cluster_metrics(y, km.labels_, Xb))

    tfidf = TfidfVectorizer(max_features=20000).fit_transform(
        [r["text"] for r in rows])
    km = KMeans(n_clusters=K, random_state=SEED, n_init=10).fit(tfidf)
    save_result("kmeans_tfidf_oracleK", cluster_metrics(y, km.labels_, None))


# ---------------- GCN ----------------

def build_file_graphs(rows, embs, use_reply, ann_dir):
    """Group rows per file; build normalized adjacency; slot labels by first appearance."""
    import torch
    by_file = defaultdict(list)
    for i, r in enumerate(rows):
        by_file[r["file"]].append(i)
    graphs = []
    for fname, idxs in sorted(by_file.items()):
        idxs = sorted(idxs, key=lambda i: rows[i]["line"])
        n = len(idxs)
        line_pos = {rows[i]["line"]: k for k, i in enumerate(idxs)}
        edges = set()
        for k in range(n - 1):  # temporal adjacency
            edges.add((k, k + 1))
        by_nick = defaultdict(list)
        for k, i in enumerate(idxs):
            by_nick[rows[i]["nick"]].append(k)
        for nick, ks in by_nick.items():  # speaker continuity
            for a in range(len(ks)):
                for b in range(a + 1, len(ks)):
                    edges.add((ks[a], ks[b]))
        if use_reply:
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
        # symmetric normalized adjacency with self-loops
        src = [a for a, b in edges] + [b for a, b in edges] + list(range(n))
        dst = [b for a, b in edges] + [a for a, b in edges] + list(range(n))
        vals = np.ones(len(src), dtype=np.float32)
        deg = np.zeros(n, dtype=np.float32)
        for s in src:
            deg[s] += 1.0
        norm = 1.0 / np.sqrt(deg)
        w = norm[src] * norm[dst] * vals
        A = torch.sparse_coo_tensor(
            np.vstack([src, dst]), w, (n, n)).coalesce()
        # slot labels: threads ordered by first appearance in this file
        slot, labels = {}, []
        for i in idxs:
            c = rows[i]["cluster"]
            if c not in slot:
                slot[c] = len(slot)
            labels.append(slot[c])
        graphs.append({"file": fname, "idxs": idxs, "A": A,
                       "X": embs[idxs], "y": np.array(labels), "K": len(slot)})
    return graphs


def simple_features(rows, idxs):
    """Handcrafted node features per the paper's 'simple features' GNN."""
    n = len(idxs)
    feats = np.zeros((n, 40), dtype=np.float32)
    for k, i in enumerate(idxs):
        r = rows[i]
        feats[k, 0] = k / max(1, n - 1)
        feats[k, 1] = len(r["text"]) / 200.0
        feats[k, 2] = len(r["text"].split()) / 40.0
        feats[k, 3] = 1.0 if "?" in r["text"] else 0.0
        feats[k, 4 + hash(r["nick"]) % 36] = 1.0
    return feats


def run_gcn(graphs_train, graphs_eval, in_dim, max_slots, tag, epochs=50):
    import torch
    import torch.nn as nn
    torch.manual_seed(SEED)
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
        order = np.random.permutation(len(graphs_train))
        for gi in order:
            g = graphs_train[gi]
            opt.zero_grad()
            logits, _ = model(g["At"], g["Xt"])
            loss = lossf(logits, g["yt"])
            loss.backward()
            opt.step()
            total += float(loss)
        if ep % 10 == 0 or ep == epochs - 1:
            print(f"[gcn:{tag}] epoch {ep} loss {total/len(graphs_train):.4f}",
                  flush=True)

    from sklearn.cluster import KMeans
    model.eval()
    all_true, pred_oracle, pred_argmax, all_emb, correct, count = [], [], [], [], 0, 0
    with torch.no_grad():
        for g in graphs_eval:
            logits, hidden = model(g["At"], g["Xt"])
            am = logits.argmax(1).cpu().numpy()
            correct += int((am == np.minimum(g["y"], max_slots - 1)).sum())
            count += len(am)
            H = hidden.cpu().numpy()
            all_emb.append(H)
            km = KMeans(n_clusters=g["K"], random_state=SEED, n_init=10).fit(H)
            all_true += [f'{g["file"]}:{c}' for c in g["y"]]
            pred_oracle += [f'{g["file"]}:{c}' for c in km.labels_]
            pred_argmax += [f'{g["file"]}:{c}' for c in am]
    X = np.vstack(all_emb)
    res = {"node_classification_accuracy": round(correct / count, 4),
           "cluster_hidden_oracleK": cluster_metrics(all_true, pred_oracle, X),
           "cluster_argmax": cluster_metrics(all_true, pred_argmax, X)}
    for g in graphs_train + graphs_eval:
        del g["At"], g["Xt"], g["yt"]
    save_result(tag, res)


def phase_gcn():
    rows_tr = load_split("train")
    rows_te = load_split("test")
    emb_tr = np.load(os.path.join(ART, "emb_large_train.npy"))
    emb_te = np.load(os.path.join(ART, "emb_large_test.npy"))
    tr_dir, te_dir = os.path.join(DATA, "train"), os.path.join(DATA, "test")

    g_tr = build_file_graphs(rows_tr, emb_tr, True, tr_dir)
    max_slots = min(200, max(g["K"] for g in g_tr))
    print(f"[gcn] max_slots={max_slots}, train graphs={len(g_tr)}")

    g_te_leak = build_file_graphs(rows_te, emb_te, True, te_dir)
    run_gcn(g_tr, g_te_leak, 1024, max_slots, "hybrid_gcn_bert_WITH_gold_reply_edges")

    g_te = build_file_graphs(rows_te, emb_te, False, te_dir)
    g_tr_nr = build_file_graphs(rows_tr, emb_tr, False, tr_dir)
    run_gcn(g_tr_nr, g_te, 1024, max_slots, "hybrid_gcn_bert_NO_reply_edges")

    Xs_tr = [simple_features(rows_tr, g["idxs"]) for g in g_tr]
    Xs_te = [simple_features(rows_te, g["idxs"]) for g in g_te_leak]
    for g, xs in zip(g_tr, Xs_tr):
        g["X"] = xs
    for g, xs in zip(g_te_leak, Xs_te):
        g["X"] = xs
    run_gcn(g_tr, g_te_leak, 40, max_slots, "gnn_simple_features_WITH_gold_reply_edges")


# ---------------- Siamese ----------------

def sample_pairs(rows, n_pairs, rng):
    by_file = defaultdict(lambda: defaultdict(list))
    for i, r in enumerate(rows):
        by_file[r["file"]][r["cluster"]].append(i)
    pairs, labels = [], []
    files = [f for f, cl in by_file.items() if len(cl) > 1]
    while len(pairs) < n_pairs:
        f = files[rng.integers(len(files))]
        clusters = [c for c, v in by_file[f].items()]
        if len(pairs) % 2 == 0:  # positive
            big = [c for c in clusters if len(by_file[f][c]) > 1]
            if not big:
                continue
            c = big[rng.integers(len(big))]
            a, b = rng.choice(by_file[f][c], 2, replace=False)
            pairs.append((a, b)); labels.append(1)
        else:  # negative
            c1, c2 = rng.choice(len(clusters), 2, replace=False)
            a = by_file[f][clusters[c1]][rng.integers(len(by_file[f][clusters[c1]]))]
            b = by_file[f][clusters[c2]][rng.integers(len(by_file[f][clusters[c2]]))]
            pairs.append((a, b)); labels.append(0)
    return np.array(pairs), np.array(labels, dtype=np.float32)


def phase_siamese():
    import torch
    import torch.nn as nn
    from sklearn.metrics import roc_auc_score
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    rows_tr, rows_te = load_split("train"), load_split("test")
    E_tr = np.load(os.path.join(ART, "emb_large_train.npy"))
    E_te = np.load(os.path.join(ART, "emb_large_test.npy"))

    tr_pairs, tr_y = sample_pairs(rows_tr, 200_000, rng)
    te_pairs, te_y = sample_pairs(rows_te, 20_000, rng)

    head = nn.Sequential(nn.Linear(2048, 256), nn.ReLU(), nn.Dropout(0.3),
                         nn.Linear(256, 1)).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=1e-3)
    lossf = nn.BCEWithLogitsLoss()

    def feats(E, pairs):
        a, b = E[pairs[:, 0]], E[pairs[:, 1]]
        return np.hstack([np.abs(a - b), a * b]).astype(np.float32)

    Xtr, Xte = feats(E_tr, tr_pairs), feats(E_te, te_pairs)
    Xtr_t = torch.tensor(Xtr)
    ytr_t = torch.tensor(tr_y)
    for ep in range(5):
        perm = torch.randperm(len(Xtr_t))
        total = 0.0
        for i in range(0, len(perm), 512):
            idx = perm[i:i + 512]
            opt.zero_grad()
            out = head(Xtr_t[idx].to(device)).squeeze(1)
            loss = lossf(out, ytr_t[idx].to(device))
            loss.backward(); opt.step(); total += float(loss) * len(idx)
        print(f"[siamese] epoch {ep} loss {total/len(perm):.4f}", flush=True)

    head.eval()
    with torch.no_grad():
        logits = head(torch.tensor(Xte).to(device)).squeeze(1).cpu().numpy()
    probs = 1 / (1 + np.exp(-logits))
    pred = (probs > 0.5).astype(int)
    tp = int(((pred == 1) & (te_y == 1)).sum()); fn = int(((pred == 0) & (te_y == 1)).sum())
    fp = int(((pred == 1) & (te_y == 0)).sum()); tn = int(((pred == 0) & (te_y == 0)).sum())
    acc = (tp + tn) / len(te_y)
    rec = tp / max(1, tp + fn)
    auc = roc_auc_score(te_y, probs)

    # thread reconstruction: link prediction within +-50-line window, connected components
    by_file = defaultdict(list)
    for i, r in enumerate(rows_te):
        by_file[r["file"]].append(i)
    all_true, all_pred = [], []
    with torch.no_grad():
        for f, idxs in sorted(by_file.items()):
            idxs = sorted(idxs, key=lambda i: rows_te[i]["line"])
            cand = [(a, b) for ai, a in enumerate(idxs)
                    for b in idxs[ai + 1: ai + 51]]
            uf = UnionFind()
            for i in range(0, len(cand), 4096):
                chunk = np.array(cand[i:i + 4096])
                Xc = torch.tensor(feats(E_te, chunk)).to(device)
                pc = torch.sigmoid(head(Xc).squeeze(1)).cpu().numpy()
                for (a, b), p in zip(chunk, pc):
                    if p > 0.5:
                        uf.union(int(a), int(b))
            all_true += [rows_te[i]["cluster"] for i in idxs]
            all_pred += [f"{f}:{uf.find(i)}" for i in idxs]
    cm = cluster_metrics(all_true, all_pred, None)
    save_result("siamese_bert_large", {
        "link_accuracy": round(acc, 4), "link_recall": round(rec, 4),
        "link_roc_auc": round(auc, 4),
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "thread_reconstruction_connected_components": cm})


def phase_triplet():
    import torch
    import torch.nn as nn
    from sklearn.cluster import KMeans
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    rows_tr, rows_te = load_split("train"), load_split("test")
    E_tr = np.load(os.path.join(ART, "emb_base_train.npy"))
    E_te = np.load(os.path.join(ART, "emb_base_test.npy"))

    pairs, ylab = sample_pairs(rows_tr, 240_000, rng)  # reuse sampler: pos rows -> anchors
    pos = pairs[ylab == 1]
    neg = pairs[ylab == 0]
    n = min(len(pos), len(neg))
    anchors, positives, negatives = pos[:n, 0], pos[:n, 1], neg[:n, 0]

    proj = nn.Linear(768, 256).to(device)
    opt = torch.optim.Adam(proj.parameters(), lr=1e-3)
    tl = nn.TripletMarginLoss(margin=1.0)
    A = torch.tensor(E_tr[anchors]); P = torch.tensor(E_tr[positives])
    N = torch.tensor(E_tr[negatives])
    for ep in range(5):
        perm = torch.randperm(n)
        total = 0.0
        for i in range(0, n, 512):
            idx = perm[i:i + 512]
            opt.zero_grad()
            loss = tl(proj(A[idx].to(device)), proj(P[idx].to(device)),
                      proj(N[idx].to(device)))
            loss.backward(); opt.step(); total += float(loss) * len(idx)
        print(f"[triplet] epoch {ep} loss {total/n:.4f}", flush=True)

    proj.eval()
    with torch.no_grad():
        Z = proj(torch.tensor(E_te).to(device)).cpu().numpy()
    y = [r["cluster"] for r in rows_te]
    K = len(set(y))
    km = KMeans(n_clusters=K, random_state=SEED, n_init=10).fit(Z)
    full = cluster_metrics(y, km.labels_, Z)

    idx1k = rng.choice(len(rows_te), 1000, replace=False)
    y1k = [y[i] for i in idx1k]
    K1k = len(set(y1k))
    km1k = KMeans(n_clusters=K1k, random_state=SEED, n_init=10).fit(Z[idx1k])
    small = cluster_metrics(y1k, km1k.labels_, Z[idx1k])
    save_result("triplet_bert_base", {"full_test_oracleK": full,
                                      "sample_1000_oracleK": small})


def phase_report():
    with open(os.path.join(ART, f"results_seed{SEED}.json")) as f:
        results = json.load(f)
    print(json.dumps(results, indent=2))


def phase_aggregate():
    """Combine all results_seed*.json into mean+-std per metric."""
    import glob
    files = sorted(glob.glob(os.path.join(ART, "results_seed*.json")))
    seeds_data = {}
    for fp in files:
        seed = os.path.basename(fp).replace("results_seed", "").replace(".json", "")
        with open(fp) as f:
            seeds_data[seed] = json.load(f)
    print(f"[aggregate] {len(seeds_data)} seeds: {list(seeds_data)}")

    def walk(d, prefix=()):
        for k, v in d.items():
            if isinstance(v, dict):
                yield from walk(v, prefix + (k,))
            elif isinstance(v, (int, float)):
                yield prefix + (k,), v

    by_path = defaultdict(list)
    for seed, data in seeds_data.items():
        for path, val in walk(data):
            by_path[path].append(val)

    summary = {}
    for path, vals in sorted(by_path.items()):
        arr = np.array(vals, dtype=float)
        summary[".".join(path)] = {
            "mean": round(float(arr.mean()), 4),
            "std": round(float(arr.std(ddof=1)) if len(arr) > 1 else 0.0, 4),
            "n_seeds": len(arr), "values": [round(v, 4) for v in vals],
        }
    with open(os.path.join(ART, "results_aggregate.json"), "w") as f:
        json.dump(summary, f, indent=2)
    for path, s in summary.items():
        print(f"[agg] {path}: {s['mean']:.4f} +/- {s['std']:.4f} (n={s['n_seeds']})")


PHASES = {"parse": phase_parse, "embed": phase_embed, "baseline": phase_baseline,
          "gcn": phase_gcn, "siamese": phase_siamese, "triplet": phase_triplet,
          "report": phase_report, "aggregate": phase_aggregate}

if __name__ == "__main__":
    args = sys.argv[1:]
    if not args or args == ["all"]:
        order = list(PHASES)
    else:
        order = args
    for name in order:
        print(f"\n===== PHASE {name} =====", flush=True)
        t0 = time.time()
        PHASES[name]()
        print(f"===== {name} done in {time.time()-t0:.0f}s =====", flush=True)
