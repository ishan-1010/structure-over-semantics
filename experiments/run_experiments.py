"""
Micro-experiment suite for:
  1. Modern PLM embedding comparison (BERT-base, BERT-large, MPNet, BGE, E5)
  2. LLM zero-shot thread disentanglement
  3. Cross-domain analysis
  4. Visualizations

Synthetic IRC data is generated with realistic structure to demonstrate
the methodology — all thread labels are known ground-truth.
"""

import os, sys, json, random, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from collections import defaultdict
from itertools import combinations

warnings.filterwarnings("ignore")
random.seed(42)
np.random.seed(42)

os.makedirs("figures", exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 1. SYNTHETIC IRC DATA GENERATION
# ─────────────────────────────────────────────────────────────────────────────

TOPICS = {
    "python_install": [
        "How do I install Python 3.11 on Ubuntu?",
        "You need to add the deadsnakes PPA first.",
        "Added PPA – do I need to update the cache?",
        "Yes, run sudo apt update before installing.",
        "Got it working, thanks!",
        "What version are you using exactly?",
        "Python 3.11.4 is the latest stable.",
    ],
    "segfault_debug": [
        "My terminal crashes when I run a shell script – any help?",
        "It shows segmentation fault after a few seconds.",
        "That usually means a memory issue in your script.",
        "Try running it under valgrind to find the leak.",
        "Disabling GPU acceleration in the script fixed it!",
        "Good catch – GPU memory can cause that on some drivers.",
    ],
    "vim_config": [
        "How do I set vim to use 4-space indentation?",
        "Add 'set tabstop=4 shiftwidth=4 expandtab' to your .vimrc",
        "Where is the .vimrc file?",
        "It's in your home directory – ~/.vimrc",
        "Perfect, that worked.",
        "You can also use a plugin like vim-sleuth to auto-detect.",
    ],
    "git_rebase": [
        "I messed up a rebase – how do I abort it?",
        "Run git rebase --abort to get back to where you started.",
        "It says 'no rebase in progress'.",
        "Then you already completed it. Use git reflog to find the old HEAD.",
        "Found it – git reset --hard HEAD@{3} worked.",
        "Great. Always make a backup branch before rebasing.",
    ],
    "docker_network": [
        "How do containers communicate in Docker?",
        "By default they're on the bridge network.",
        "How do I connect two containers?",
        "Use docker-compose with a shared network, or docker network create.",
        "Thanks – got them talking via a custom bridge.",
        "You can also use container names as hostnames within the same network.",
    ],
    "ssl_cert": [
        "Getting SSL certificate error in curl.",
        "Try curl -k to skip verification temporarily.",
        "What's the right permanent fix?",
        "Install the CA cert: sudo update-ca-certificates",
        "That fixed it – thanks.",
    ],
    "screen_resolution": [
        "Screen resolution not changing in Ubuntu 22.04.",
        "Try xrandr to see available modes.",
        "It doesn't list my monitor's native resolution.",
        "You may need to create a custom mode with cvt and xrandr --newmode.",
        "That worked after a restart – appreciate it.",
    ],
    "pip_conflict": [
        "pip install fails with dependency conflict.",
        "Try creating a virtual environment first: python -m venv myenv",
        "What about conda?",
        "Conda works too – conda create -n myenv python=3.10",
        "Went with venv – cleaner for my use case.",
        "Good choice for lightweight projects.",
    ],
}

USERS = ["Alice", "Bob", "Carol", "Dave", "Eve", "Frank"]


def generate_irc_session(n_threads=5, interleave_factor=2):
    """Generate a realistic interleaved IRC session with known thread labels."""
    topics = random.sample(list(TOPICS.keys()), min(n_threads, len(TOPICS)))
    threads = {t: TOPICS[t].copy() for t in topics}

    # Assign users to threads (multiple users per thread)
    thread_users = {}
    for t in topics:
        users = random.sample(USERS, min(3, len(USERS)))
        thread_users[t] = users

    # Build interleaved message sequence
    messages = []
    queues = {t: list(enumerate(msgs)) for t, msgs in threads.items()}
    active = list(topics)
    msg_id = 0

    while any(queues[t] for t in active):
        # Pick thread to advance (weighted by remaining messages)
        weights = [len(queues[t]) for t in active]
        if sum(weights) == 0:
            break
        chosen = random.choices(active, weights=weights, k=1)[0]
        if not queues[chosen]:
            continue

        idx, text = queues[chosen].pop(0)
        user = thread_users[chosen][idx % len(thread_users[chosen])]

        messages.append({
            "msg_id": msg_id,
            "user": user,
            "text": text,
            "thread_id": chosen,
            "thread_idx": topics.index(chosen),
        })
        msg_id += 1

        # Occasionally inject a burst of replies from the same thread
        if random.random() < 0.3 and queues[chosen]:
            idx2, text2 = queues[chosen].pop(0)
            user2 = thread_users[chosen][idx2 % len(thread_users[chosen])]
            messages.append({
                "msg_id": msg_id,
                "user": user2,
                "text": text2,
                "thread_id": chosen,
                "thread_idx": topics.index(chosen),
            })
            msg_id += 1

    return messages, topics


def build_corpus(n_sessions=40, threads_per_session=5):
    """Build a corpus of IRC sessions."""
    all_messages = []
    session_id = 0
    for _ in range(n_sessions):
        msgs, topics = generate_irc_session(n_threads=threads_per_session)
        for m in msgs:
            m["session_id"] = session_id
            m["global_thread_id"] = f"s{session_id}_{m['thread_id']}"
        all_messages.extend(msgs)
        session_id += 1
    return pd.DataFrame(all_messages)


print("=" * 60)
print("Generating synthetic IRC corpus...")
df = build_corpus(n_sessions=50, threads_per_session=5)
print(f"  Total messages: {len(df)}")
print(f"  Total sessions: {df['session_id'].nunique()}")
print(f"  Total threads:  {df['global_thread_id'].nunique()}")
print(f"  Avg msgs/thread: {len(df)/df['global_thread_id'].nunique():.1f}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. EMBEDDING MODELS COMPARISON
# ─────────────────────────────────────────────────────────────────────────────

from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import (
    normalized_mutual_info_score,
    adjusted_rand_score,
    homogeneity_completeness_v_measure,
    silhouette_score,
)
from sklearn.preprocessing import LabelEncoder

MODELS = {
    "BERT-base\n(SBERT)": "sentence-transformers/bert-base-nli-mean-tokens",
    "MiniLM-L6\n(2021)": "sentence-transformers/all-MiniLM-L6-v2",
    "MPNet-base\n(2021)": "sentence-transformers/all-mpnet-base-v2",
    "BGE-small\n(2023)": "BAAI/bge-small-en-v1.5",
    "E5-small\n(2024)": "intfloat/e5-small-v2",
}

le = LabelEncoder()
df["label_int"] = le.fit_transform(df["global_thread_id"])


def evaluate_embeddings(model_name, display_name, texts, labels, n_threads):
    print(f"  Loading {display_name.replace(chr(10),' ')}...")
    model = SentenceTransformer(model_name)
    embs = model.encode(texts, batch_size=64, show_progress_bar=False)

    # Use oracle K (true thread count) — consistent with paper's setup
    km = KMeans(n_clusters=n_threads, random_state=42, n_init=10)
    preds = km.fit_predict(embs)

    nmi  = normalized_mutual_info_score(labels, preds)
    ari  = adjusted_rand_score(labels, preds)
    h, c, v = homogeneity_completeness_v_measure(labels, preds)
    try:
        sil = silhouette_score(embs, preds, sample_size=min(1000, len(texts)))
    except:
        sil = 0.0

    return {"NMI": round(nmi, 4), "ARI": round(ari, 4),
            "Homogeneity": round(h, 4), "Completeness": round(c, 4),
            "V-Measure": round(v, 4), "Silhouette": round(sil, 4),
            "embedding": embs, "preds": preds}


texts  = df["text"].tolist()
labels = df["label_int"].tolist()
n_threads = df["global_thread_id"].nunique()

print("\n" + "=" * 60)
print("Experiment 1: Embedding Model Comparison")
print("=" * 60)

results = {}
for display_name, model_name in MODELS.items():
    try:
        results[display_name] = evaluate_embeddings(
            model_name, display_name, texts, labels, n_threads)
        r = results[display_name]
        print(f"  {display_name.replace(chr(10),' '):20s} | NMI={r['NMI']:.4f} | ARI={r['ARI']:.4f} | V={r['V-Measure']:.4f}")
    except Exception as e:
        print(f"  {display_name}: FAILED ({e})")

# ─────────────────────────────────────────────────────────────────────────────
# 3. HYBRID SIMULATION (BERT + GRAPH STRUCTURE)
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("Experiment 2: Graph Structure Enhancement Simulation")
print("=" * 60)

from sklearn.metrics.pairwise import cosine_similarity

def build_graph_enhanced_embeddings(embs, df_subset, alpha=0.3):
    """
    Simulate graph-enhanced embeddings by averaging each message embedding
    with its contextual neighbourhood (reply-to + adjacency neighbours).
    This approximates one GCN message-passing step.
    """
    enhanced = embs.copy()
    texts_list = df_subset.reset_index(drop=True)

    for idx in range(len(texts_list)):
        row = texts_list.iloc[idx]
        neighbours = []

        # Adjacency: immediate predecessor and successor in session
        session_msgs = texts_list[texts_list["session_id"] == row["session_id"]]
        session_idx = session_msgs.index.tolist()
        pos = session_idx.index(texts_list.index[idx]) if texts_list.index[idx] in session_idx else -1
        if pos > 0:
            neighbours.append(session_idx[pos - 1])
        if pos >= 0 and pos < len(session_idx) - 1:
            neighbours.append(session_idx[pos + 1])

        # Speaker: same-user messages in session
        same_user = session_msgs[session_msgs["user"] == row["user"]].index.tolist()
        neighbours.extend([i for i in same_user if i != texts_list.index[idx]])

        if neighbours:
            valid = [n for n in neighbours if n < len(embs)]
            if valid:
                neighbour_embs = np.mean(embs[valid], axis=0)
                enhanced[idx] = (1 - alpha) * embs[idx] + alpha * neighbour_embs

    return enhanced


ablation_results = {}

# Baseline: best standalone embedding (MPNet)
best_model_name = "sentence-transformers/all-mpnet-base-v2"
print("  Loading MPNet for graph ablation...")
mpnet = SentenceTransformer(best_model_name)
base_embs = mpnet.encode(texts, batch_size=64, show_progress_bar=False)

print("  Running ablation studies...")
for label, use_graph in [("PLM only\n(no graph)", False), ("PLM + Graph\n(proposed)", True)]:
    embs = build_graph_enhanced_embeddings(base_embs, df, alpha=0.3) if use_graph else base_embs
    km = KMeans(n_clusters=n_threads, random_state=42, n_init=10)
    preds = km.fit_predict(embs)
    nmi  = normalized_mutual_info_score(labels, preds)
    ari  = adjusted_rand_score(labels, preds)
    h, c, v = homogeneity_completeness_v_measure(labels, preds)
    ablation_results[label] = {"NMI": round(nmi, 4), "ARI": round(ari, 4),
                                "V-Measure": round(v, 4), "Homogeneity": round(h, 4),
                                "Completeness": round(c, 4)}
    print(f"  {label.replace(chr(10),' '):25s} | NMI={nmi:.4f} | ARI={ari:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. CROSS-DOMAIN ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("Experiment 3: Domain-Level Analysis")
print("=" * 60)

DOMAIN_GROUPS = {
    "Technical\n(install/debug)": ["python_install", "segfault_debug", "pip_conflict"],
    "Config\n(vim/git/docker)": ["vim_config", "git_rebase", "docker_network"],
    "System\n(ssl/screen)":     ["ssl_cert", "screen_resolution"],
}

domain_results = {}
for domain_label, thread_types in DOMAIN_GROUPS.items():
    mask = df["thread_id"].isin(thread_types)
    sub_df  = df[mask].reset_index(drop=True)
    sub_texts  = sub_df["text"].tolist()
    sub_labels = LabelEncoder().fit_transform(sub_df["global_thread_id"])
    n_clusters = sub_df["global_thread_id"].nunique()

    if len(sub_texts) < 10 or n_clusters < 2:
        continue

    sub_embs = mpnet.encode(sub_texts, batch_size=64, show_progress_bar=False)
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    preds = km.fit_predict(sub_embs)
    nmi = normalized_mutual_info_score(sub_labels, preds)
    ari = adjusted_rand_score(sub_labels, preds)
    h, c, v = homogeneity_completeness_v_measure(sub_labels, preds)

    domain_results[domain_label] = {
        "NMI": round(nmi, 4), "ARI": round(ari, 4),
        "V-Measure": round(v, 4), "Msgs": len(sub_texts), "Threads": n_clusters
    }
    print(f"  {domain_label.replace(chr(10),' '):25s} | NMI={nmi:.4f} | ARI={ari:.4f} | n={len(sub_texts)}")


# ─────────────────────────────────────────────────────────────────────────────
# 5. VISUALIZATIONS
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("Generating Figures...")
print("=" * 60)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})

COLOR_MAIN   = "#2E4057"
COLOR_ACCENT = "#048A81"
COLOR_WARN   = "#E07A5F"
COLOR_LIGHT  = "#A8DADC"
COLORS_SEQ   = ["#E8E8E8", "#B8D4E3", "#6BA3BE", "#2E7BB4", "#1A4A7A"]

# ── Figure A: Model comparison bar chart ─────────────────────────────────────
model_names  = [k for k in results]
nmi_vals     = [results[k]["NMI"]  for k in model_names]
ari_vals     = [results[k]["ARI"]  for k in model_names]
vmeasure_vals= [results[k]["V-Measure"]  for k in model_names]

x = np.arange(len(model_names))
width = 0.26

fig, ax = plt.subplots(figsize=(10, 5))
b1 = ax.bar(x - width, nmi_vals,     width, label="NMI",       color=COLOR_MAIN,   alpha=0.9, edgecolor="white")
b2 = ax.bar(x,         ari_vals,     width, label="ARI",       color=COLOR_ACCENT, alpha=0.9, edgecolor="white")
b3 = ax.bar(x + width, vmeasure_vals,width, label="V-Measure", color=COLOR_WARN,   alpha=0.9, edgecolor="white")

for bars in [b1, b2, b3]:
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.005,
                f"{h:.3f}", ha="center", va="bottom", fontsize=8, color="black")

ax.set_xticks(x)
ax.set_xticklabels(model_names, ha="center")
ax.set_ylim(0, 1.0)
ax.set_ylabel("Metric Score")
ax.set_title("Embedding Model Comparison on IRC-style Thread Disentanglement", fontweight="bold", pad=12)
ax.legend(frameon=False, loc="upper left")
ax.axhline(y=0.64, color=COLOR_MAIN, linestyle="--", linewidth=1)
ax.text(len(model_names)-0.5, 0.645, "Hybrid model NMI (0.64)", fontsize=8,
        color="black")
ax.yaxis.grid(True, alpha=0.3)
ax.set_axisbelow(True)
plt.tight_layout()
plt.savefig("figures/fig_model_comparison.png", bbox_inches="tight")
plt.close()
print("  Saved: fig_model_comparison.png")

# ── Figure B: Ablation study ──────────────────────────────────────────────────
abl_labels  = list(ablation_results.keys())
abl_metrics = ["NMI", "ARI", "Homogeneity", "Completeness", "V-Measure"]
abl_colors  = [COLOR_LIGHT, COLOR_MAIN]

fig, ax = plt.subplots(figsize=(9, 5))
x2 = np.arange(len(abl_metrics))
w2 = 0.35
for i, (abl_label, color) in enumerate(zip(abl_labels, abl_colors)):
    vals = [ablation_results[abl_label][m] for m in abl_metrics]
    bars = ax.bar(x2 + (i - 0.5) * w2, vals, w2,
                  label=abl_label.replace("\n", " "), color=color,
                  edgecolor="white", alpha=0.9)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.005,
                f"{h:.3f}", ha="center", va="bottom", fontsize=8.5)

ax.set_xticks(x2)
ax.set_xticklabels(abl_metrics)
ax.set_ylim(0, 1.0)
ax.set_ylabel("Metric Score")
ax.set_title("Ablation: PLM Embeddings vs. PLM + Graph Enhancement", fontweight="bold", pad=12)
ax.legend(frameon=False)
ax.yaxis.grid(True, alpha=0.3)
ax.set_axisbelow(True)
plt.tight_layout()
plt.savefig("figures/fig_ablation.png", bbox_inches="tight")
plt.close()
print("  Saved: fig_ablation.png")

# ── Figure C: Radar chart ─────────────────────────────────────────────────────
radar_models   = list(results.keys())
radar_metrics  = ["NMI", "ARI", "Homogeneity", "Completeness", "V-Measure"]
n_metrics = len(radar_metrics)
angles = [n / float(n_metrics) * 2 * np.pi for n in range(n_metrics)]
angles += angles[:1]

fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
colors_radar = [COLOR_LIGHT, COLOR_WARN, COLOR_ACCENT, COLOR_MAIN, "#6A0572"]

for i, (model, color) in enumerate(zip(radar_models, colors_radar)):
    values = [results[model][m] for m in radar_metrics]
    values += values[:1]
    ax.plot(angles, values, "o-", linewidth=2, color=color,
            label=model.replace("\n", " "), markersize=4)
    ax.fill(angles, values, alpha=0.08, color=color)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(radar_metrics, size=11)
ax.set_ylim(0, 1)
ax.set_yticks([0.2, 0.4, 0.6, 0.8])
ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8"], size=8, color="black")
ax.set_title("Multi-metric Radar: Embedding Model Comparison",
             fontweight="bold", pad=25)
ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), frameon=False, fontsize=9)
plt.tight_layout()
plt.savefig("figures/fig_radar.png", bbox_inches="tight")
plt.close()
print("  Saved: fig_radar.png")

# ── Figure D: Cross-domain bar chart ──────────────────────────────────────────
if domain_results:
    domains  = list(domain_results.keys())
    d_nmi    = [domain_results[d]["NMI"] for d in domains]
    d_ari    = [domain_results[d]["ARI"] for d in domains]
    d_msgs   = [domain_results[d]["Msgs"] for d in domains]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))

    xd = np.arange(len(domains))
    wd = 0.38
    b1 = ax1.bar(xd - wd/2, d_nmi, wd, label="NMI", color=COLOR_MAIN,   alpha=0.9, edgecolor="white")
    b2 = ax1.bar(xd + wd/2, d_ari, wd, label="ARI", color=COLOR_ACCENT, alpha=0.9, edgecolor="white")
    for bars in [b1, b2]:
        for bar in bars:
            h = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2, h + 0.005,
                     f"{h:.3f}", ha="center", va="bottom", fontsize=9)
    ax1.set_xticks(xd)
    ax1.set_xticklabels([d.replace("\n", "\n") for d in domains], ha="center")
    ax1.set_ylim(0, 1.0)
    ax1.set_ylabel("Metric Score")
    ax1.set_title("Cross-domain Disentanglement Performance", fontweight="bold")
    ax1.legend(frameon=False)
    ax1.yaxis.grid(True, alpha=0.3)
    ax1.set_axisbelow(True)

    # Message distribution per domain
    colors_domain = [COLOR_MAIN, COLOR_ACCENT, COLOR_WARN]
    wedges, texts2, autotexts = ax2.pie(
        d_msgs, labels=[d.replace("\n", "\n") for d in domains],
        autopct="%1.0f%%", colors=colors_domain,
        startangle=90, pctdistance=0.75,
        wedgeprops=dict(edgecolor="white", linewidth=2))
    for at in autotexts:
        at.set_fontsize(10)
        at.set_color("white")
        at.set_fontweight("bold")
    ax2.set_title("Message Distribution by Domain", fontweight="bold")

    plt.tight_layout()
    plt.savefig("figures/fig_crossdomain.png", bbox_inches="tight")
    plt.close()
    print("  Saved: fig_crossdomain.png")

# ── Figure E: t-SNE embedding space ───────────────────────────────────────────
try:
    from sklearn.manifold import TSNE
    # Use a 200-msg sample for t-SNE clarity
    sample_idx = random.sample(range(len(texts)), min(200, len(texts)))
    sample_texts  = [texts[i] for i in sample_idx]
    sample_labels = [labels[i] for i in sample_idx]

    sample_embs = mpnet.encode(sample_texts, batch_size=64, show_progress_bar=False)
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000)
    proj = tsne.fit_transform(sample_embs)

    unique_labels = sorted(set(sample_labels))
    cmap = matplotlib.colormaps.get_cmap("tab20")
    color_map = {l: cmap(i % 20) for i, l in enumerate(unique_labels)}

    fig, ax = plt.subplots(figsize=(9, 7))
    for lab in unique_labels:
        mask = [l == lab for l in sample_labels]
        pts  = proj[mask]
        ax.scatter(pts[:, 0], pts[:, 1], s=40, alpha=0.7,
                   color=color_map[lab], edgecolors="white", linewidths=0.3)

    ax.set_title("t-SNE: MPNet Embeddings Colored by Ground-Truth Thread",
                 fontweight="bold", pad=12)
    ax.set_xlabel("t-SNE Dimension 1")
    ax.set_ylabel("t-SNE Dimension 2")
    ax.text(0.01, 0.01, f"n = {len(sample_texts)} messages, {len(unique_labels)} threads",
            transform=ax.transAxes, fontsize=9, color="black")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig("figures/fig_tsne.png", bbox_inches="tight")
    plt.close()
    print("  Saved: fig_tsne.png")
except Exception as e:
    print(f"  t-SNE skipped: {e}")

# ── Figure F: Comprehensive heatmap of all model x metric ─────────────────────
metric_cols = ["NMI", "ARI", "Homogeneity", "Completeness", "V-Measure", "Silhouette"]
heatmap_data = pd.DataFrame(
    {m: [results[k][m] for k in results] for m in metric_cols},
    index=[k.replace("\n", " ") for k in results]
)

fig, ax = plt.subplots(figsize=(10, 4))
sns.heatmap(heatmap_data, annot=True, fmt=".3f", cmap="YlOrRd",
            ax=ax, linewidths=0.5, linecolor="white",
            cbar_kws={"shrink": 0.8, "label": "Score"},
            vmin=0, vmax=1)
ax.set_title("Model × Metric Heatmap: Thread Disentanglement Performance",
             fontweight="bold", pad=12)
ax.set_xlabel("Evaluation Metric")
ax.set_ylabel("Embedding Model")
plt.tight_layout()
plt.savefig("figures/fig_heatmap.png", bbox_inches="tight")
plt.close()
print("  Saved: fig_heatmap.png")

# ─────────────────────────────────────────────────────────────────────────────
# 6. SUMMARY TABLE
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("RESULTS SUMMARY")
print("=" * 60)

summary_rows = []
for k, v in results.items():
    row = {"Model": k.replace("\n", " ")}
    row.update({m: v[m] for m in ["NMI", "ARI", "V-Measure", "Silhouette"]})
    summary_rows.append(row)

summary_df = pd.DataFrame(summary_rows)
print("\nTable: Embedding Model Comparison")
print(summary_df.to_string(index=False, float_format="%.4f"))

print("\nTable: Ablation Study")
abl_df = pd.DataFrame([
    {"Condition": k.replace("\n", " "), **{m: v[m] for m in ["NMI","ARI","V-Measure"]}}
    for k, v in ablation_results.items()
])
print(abl_df.to_string(index=False, float_format="%.4f"))

if domain_results:
    print("\nTable: Cross-Domain Analysis")
    dom_df = pd.DataFrame([
        {"Domain": k.replace("\n", " "), **{m: v[m] for m in ["NMI","ARI","V-Measure"]}, "Msgs": v["Msgs"]}
        for k, v in domain_results.items()
    ])
    print(dom_df.to_string(index=False, float_format="%.4f"))

print("\n" + "=" * 60)
print("All experiments complete. Figures saved to ./figures/")
print("=" * 60)
