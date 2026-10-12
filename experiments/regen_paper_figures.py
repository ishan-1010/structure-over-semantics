#!/usr/bin/env python3
"""
Regenerates the four extended-analysis figures used in the paper
(figS1-figS4) from the synthetic IRC corpus, in the layouts the paper's
captions describe. Fixes vs the original generation run:
  - the model labeled "BERT-base" is a genuine BERT-base sentence encoder
    (sentence-transformers/bert-base-nli-mean-tokens), not GloVe
  - all annotation text is black (no grey), spellings are American
Deterministic: same seeds as run_experiments.py (random 42 / np 42).
"""
import random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

random.seed(42)
np.random.seed(42)

# --- synthetic corpus: identical code path to run_experiments.py ---
from run_experiments_data import TOPICS, USERS, generate_irc_session, build_corpus  # noqa

COLOR_MAIN, COLOR_ACCENT, COLOR_WARN, COLOR_LIGHT = "#2E4057", "#048A81", "#E07A5F", "#A8DADC"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                     "axes.titlesize": 13, "axes.labelsize": 11,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 150})

df = build_corpus(n_sessions=50, threads_per_session=5)
print(f"corpus: {len(df)} messages, {df['global_thread_id'].nunique()} threads")

from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import (normalized_mutual_info_score, adjusted_rand_score,
                             homogeneity_completeness_v_measure)
from sklearn.preprocessing import LabelEncoder

texts = df["text"].tolist()
labels = LabelEncoder().fit_transform(df["global_thread_id"])
K = df["global_thread_id"].nunique()

MODELS = {
    "BERT-base\n(SBERT)": "sentence-transformers/bert-base-nli-mean-tokens",
    "MiniLM-L6\n(2021)": "sentence-transformers/all-MiniLM-L6-v2",
    "MPNet-base\n(2021)": "sentence-transformers/all-mpnet-base-v2",
    "BGE-small\n(2023)": "BAAI/bge-small-en-v1.5",
    "E5-small\n(2024)": "intfloat/e5-small-v2",
}

results = {}
for name, path in MODELS.items():
    model = SentenceTransformer(path)
    embs = model.encode(texts, batch_size=64, show_progress_bar=False)
    preds = KMeans(n_clusters=K, random_state=42, n_init=10).fit_predict(embs)
    h, c, v = homogeneity_completeness_v_measure(labels, preds)
    results[name] = {"NMI": normalized_mutual_info_score(labels, preds),
                     "ARI": adjusted_rand_score(labels, preds),
                     "Homogeneity": h, "Completeness": c, "V-Measure": v,
                     "embs": embs}
    print(f"{name.replace(chr(10), ' ')}: NMI={results[name]['NMI']:.4f} "
          f"H={h:.4f} C={c:.4f}")

# ---------- figS1: embedding backbone comparison (4 metrics, no ARI) ----------
metrics = ["NMI", "Homogeneity", "Completeness", "V-Measure"]
colors = [COLOR_MAIN, COLOR_ACCENT, COLOR_WARN, COLOR_LIGHT]
names = list(results)
x = np.arange(len(names))
w = 0.2
fig, ax = plt.subplots(figsize=(11, 5.5))
for mi, (m, col) in enumerate(zip(metrics, colors)):
    vals = [results[n][m] for n in names]
    bars = ax.bar(x + (mi - 1.5) * w, vals, w, label=m, color=col,
                  edgecolor="white", alpha=0.95)
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.006,
                f"{b.get_height():.3f}", ha="center", va="bottom",
                fontsize=8, color="black")
ax.set_xticks(x)
ax.set_xticklabels(names)
ax.set_ylim(0, 0.95)
ax.set_ylabel("Score")
ax.set_title("Embedding Backbone Comparison on Thread Disentanglement\n"
             "(unsupervised K-Means, oracle K, IRC-style controlled data)",
             fontweight="bold")
# Real-corpus graph-based results (5-seed mean, official Kummerfeld test split,
# Table 3): GCN+BERT-large 0.835, GCN+simple-features 0.861. Shown as a band,
# not a single line, since the point of this figure is that embedding choice
# barely moves the number once graph structure is present.
ax.axhspan(0.835, 0.861, color=COLOR_MAIN, alpha=0.15)
ax.axhline(y=0.835, color=COLOR_MAIN, linestyle="--", linewidth=1.2)
ax.axhline(y=0.861, color=COLOR_MAIN, linestyle="--", linewidth=1.2)
ax.text(len(names) - 0.4, 0.87, "Real-corpus graph-based NMI, this work\n(0.835 BERT-fed -- 0.861 simple features)",
        fontsize=8.5, color="black", ha="right", va="bottom")
ax.legend(frameon=False, loc="upper left", ncol=4)
ax.yaxis.grid(True, alpha=0.3)
ax.set_axisbelow(True)
plt.tight_layout()
plt.savefig("figures/figS1_embedding_comparison.png", bbox_inches="tight")
plt.close()
print("saved figS1")

# ---------- figS2: ablation (MPNet +- graph), 5 metrics ----------
def graph_enhance(embs, frame, alpha=0.3):
    enhanced = embs.copy()
    fr = frame.reset_index(drop=True)
    for idx in range(len(fr)):
        row = fr.iloc[idx]
        sess = fr[fr["session_id"] == row["session_id"]]
        sidx = sess.index.tolist()
        pos = sidx.index(idx) if idx in sidx else -1
        nb = []
        if pos > 0:
            nb.append(sidx[pos - 1])
        if 0 <= pos < len(sidx) - 1:
            nb.append(sidx[pos + 1])
        nb += [i for i in sess[sess["user"] == row["user"]].index if i != idx]
        if nb:
            enhanced[idx] = (1 - alpha) * embs[idx] + alpha * np.mean(embs[nb], axis=0)
    return enhanced

mpnet_embs = results["MPNet-base\n(2021)"]["embs"]
abl = {}
for lab, use_graph in [("PLM only\n(no graph)", False), ("PLM + Graph\n(proposed)", True)]:
    e = graph_enhance(mpnet_embs, df) if use_graph else mpnet_embs
    preds = KMeans(n_clusters=K, random_state=42, n_init=10).fit_predict(e)
    h, c, v = homogeneity_completeness_v_measure(labels, preds)
    abl[lab] = {"NMI": normalized_mutual_info_score(labels, preds),
                "ARI": adjusted_rand_score(labels, preds),
                "Homogeneity": h, "Completeness": c, "V-Measure": v}
    print(f"{lab.replace(chr(10), ' ')}: NMI={abl[lab]['NMI']:.4f}")

abl_metrics = ["NMI", "ARI", "Homogeneity", "Completeness", "V-Measure"]
fig, ax = plt.subplots(figsize=(9, 5))
x2 = np.arange(len(abl_metrics))
for i, (lab, col) in enumerate(zip(abl, [COLOR_LIGHT, COLOR_MAIN])):
    vals = [abl[lab][m] for m in abl_metrics]
    bars = ax.bar(x2 + (i - 0.5) * 0.35, vals, 0.35,
                  label=lab.replace("\n", " "), color=col,
                  edgecolor="white", alpha=0.95)
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, max(b.get_height(), 0) + 0.006,
                f"{b.get_height():.3f}", ha="center", va="bottom",
                fontsize=8.5, color="black")
ax.set_xticks(x2)
ax.set_xticklabels(abl_metrics)
ax.set_ylim(0, 1.0)
ax.set_ylabel("Metric Score")
ax.set_title("Ablation: PLM Embeddings vs. PLM + Graph Enhancement",
             fontweight="bold", pad=12)
ax.legend(frameon=False)
ax.yaxis.grid(True, alpha=0.3)
ax.set_axisbelow(True)
plt.tight_layout()
plt.savefig("figures/figS2_ablation.png", bbox_inches="tight")
plt.close()
print("saved figS2")

# ---------- figS3: t-SNE of ONE session (5 threads) ----------
from sklearn.manifold import TSNE

sess = df[df["session_id"] == 0].reset_index(drop=True)
topic_names = {"python_install": "Python Install", "segfault_debug": "Segfault Debug",
               "vim_config": "Vim Config", "git_rebase": "Git Rebase",
               "docker_network": "Docker Network", "ssl_cert": "SSL Cert",
               "screen_resolution": "Screen Resolution", "pip_conflict": "Pip Conflict"}
s_embs = SentenceTransformer(MODELS["MPNet-base\n(2021)"]).encode(
    sess["text"].tolist(), batch_size=64, show_progress_bar=False)
proj = TSNE(n_components=2, perplexity=min(10, len(sess) - 1),
            random_state=42, max_iter=2000).fit_transform(s_embs)
topics = sess["thread_id"].unique().tolist()
markers = ["o", "s", "D", "^", "v", "P", "X", "*"]
palette = [COLOR_MAIN, COLOR_ACCENT, COLOR_WARN, "#F2CC8F", "#9B5DE5"]
fig, ax = plt.subplots(figsize=(9, 7))
for ti, t in enumerate(topics):
    mask = (sess["thread_id"] == t).to_numpy()
    ax.scatter(proj[mask, 0], proj[mask, 1], s=140,
               marker=markers[ti % len(markers)],
               color=palette[ti % len(palette)],
               label=topic_names.get(t, t), edgecolors="white", linewidths=0.6)
ax.set_title("t-SNE of MPNet Embeddings Colored by Conversation Thread\n"
             f"IRC-style technical support ({len(topics)} concurrent threads, "
             f"{len(sess)} messages)", fontweight="bold")
ax.set_xlabel("t-SNE Dimension 1")
ax.set_ylabel("t-SNE Dimension 2")
ax.legend(title="Thread Topic", frameon=True)
ax.text(0.01, 0.01, "Each point = one message; cluster = same thread",
        transform=ax.transAxes, fontsize=9, color="black")
plt.tight_layout()
plt.savefig("figures/figS3_tsne.png", bbox_inches="tight")
plt.close()
print(f"saved figS3 (session 0: {len(sess)} msgs, topics: "
      f"{[topic_names.get(t, t) for t in topics]})")

# ---------- figS4: cross-domain single panel ----------
DOMAIN_GROUPS = {
    "Technical\n(install/debug)": ["python_install", "segfault_debug", "pip_conflict"],
    "Config\n(vim/git/docker)": ["vim_config", "git_rebase", "docker_network"],
    "System\n(ssl/screen)": ["ssl_cert", "screen_resolution"],
}
dom = {}
for dlab, tt in DOMAIN_GROUPS.items():
    sub = df[df["thread_id"].isin(tt)].reset_index(drop=True)
    sl = LabelEncoder().fit_transform(sub["global_thread_id"])
    kk = sub["global_thread_id"].nunique()
    se = SentenceTransformer(MODELS["MPNet-base\n(2021)"]).encode(
        sub["text"].tolist(), batch_size=64, show_progress_bar=False)
    preds = KMeans(n_clusters=kk, random_state=42, n_init=10).fit_predict(se)
    dom[dlab] = {"NMI": normalized_mutual_info_score(sl, preds), "n": len(sub)}
    print(f"{dlab.replace(chr(10), ' ')}: NMI={dom[dlab]['NMI']:.4f} n={len(sub)}")

fig, ax = plt.subplots(figsize=(9, 6))
dlabs = list(dom)
vals = [dom[d]["NMI"] for d in dlabs]
bars = ax.bar(dlabs, vals, color=[COLOR_ACCENT, COLOR_MAIN, COLOR_WARN],
              width=0.55, edgecolor="white")
for b, d in zip(bars, dlabs):
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.008,
            f"NMI={b.get_height():.3f}", ha="center", va="bottom",
            fontweight="bold", color="black")
    ax.text(b.get_x() + b.get_width() / 2, 0.012, f"n={dom[d]['n']}",
            ha="center", va="bottom", color="white", fontweight="bold")
ax.set_ylim(0, 0.42)
ax.set_ylabel("NMI Score")
ax.set_title("Cross-Domain Thread Disentanglement Performance\n"
             "(MPNet embeddings, K-Means with oracle K)", fontweight="bold")
ax.annotate("System topics harder:\nshorter threads +\nspecialized jargon",
            xy=(2, vals[2]), xytext=(1.45, 0.35), fontsize=11, color="black",
            fontweight="bold",
            bbox=dict(boxstyle="round", fc="white", ec=COLOR_WARN),
            arrowprops=dict(arrowstyle="->", color=COLOR_WARN, lw=2))
ax.yaxis.grid(True, alpha=0.3)
ax.set_axisbelow(True)
plt.tight_layout()
plt.savefig("figures/figS4_crossdomain.png", bbox_inches="tight")
plt.close()
print("saved figS4")
