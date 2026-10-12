# Structure Over Semantics

Code and per-seed results for **"Structure Over Semantics: A Controlled Re-Evaluation of
Multi-Party Conversation Disentanglement"**, Ishan Katoch, ROCLING 2026 (paper 43).

## The finding

On the IRC Disentanglement Corpus under per-file oracle *K*, grouping messages by author
scores **0.4287 ARI** with no model at all. The best GCN here scores 0.4287 as well, to four
decimals, because a graph whose only edges are speaker-continuity links decomposes into
exactly one connected component per author. The model re-derives the baseline rather than
beating it.

| Configuration | ARI (per-file oracle *K*) |
|---|---|
| Group by author, no model | 0.4287 |
| GCN, speaker edges only | 0.4287 ± 0.0000 |
| GCN, handcrafted + frozen BERT-large | 0.402 ± 0.003 |
| GCN, handcrafted features | 0.401 ± 0.003 |
| GCN, frozen BERT-large [CLS] | 0.400 ± 0.003 |
| GCN, constant node features | 0.062 |

Edge ablation, handcrafted features held fixed: speaker continuity 0.4287, gold reply-to
0.275, temporal adjacency 0.125, all three together 0.401.

Handcrafted and frozen BERT-large differ by 0.0017 (paired *t*-test over five seeds,
p = 0.40). That is a null result at low power, not demonstrated equivalence. Under the
global oracle-*K* protocol the gap is larger and favours BERT (0.377 vs 0.392).

## Corpus

Kummerfeld et al. (2019) `irc-disentanglement`, official release, used as distributed with
no re-splitting: 67,463 train / 2,500 dev / 5,000 test messages, 961 test threads.
The corpus is not redistributed here. Clone it next to `experiments/`:

```
git clone https://github.com/jkkummerfeld/irc-disentanglement.git irc_data
```

The scripts expect the annotated files at `irc_data/data/`.

## Layout

Real-corpus pipeline, which is where every number above comes from:

- `experiments/reviewer_response.py` — matched feature comparison and edge ablations.
  Identical graph, training and clustering settings across conditions, features and edges
  varied one at a time. Writes `core_artifacts/reviewer_response_seed{SEED}.json`.
- `experiments/core_rerun.py` — full pipeline: parsing, frozen BERT encoding, clustering,
  the Siamese and triplet baselines.
- `experiments/rr_aggregate.py` — mean and standard deviation per configuration under both
  clustering protocols, paired significance tests across seeds, per-file stratification.
- `experiments/make_tables.py` — emits the camera-ready LaTeX tables directly from the JSON,
  so no number in the paper is transcribed by hand.

Synthetic secondary check, reported as Section 5.6 and Figure 2 of the paper and clearly
labelled as such there:

- `experiments/run_experiments.py`, `experiments/run_experiments_data.py` — generated
  IRC-style sessions with known thread labels, used to check that the NMI pattern holds
  outside the real corpus.
- `experiments/regen_paper_figures.py` — regenerates the figures in `figures/` from that
  synthetic corpus.

## Reproducing

Seeds 42 to 46, one per run:

```
cd experiments
for s in 42 43 44 45 46; do REEVAL_SEED=$s python reviewer_response.py; done
python rr_aggregate.py
python make_tables.py
```

The per-seed JSON from the runs behind the paper is already in
`experiments/core_artifacts/`, so the aggregation and tables can be checked without
re-running the models. Encoders are frozen throughout; there is no fine-tuning, so
seed-to-seed variance comes from clustering initialisation, GCN weight initialisation and
dropout, and pair sampling.

## What this does not show

One GCN, one corpus, ARI and NMI only. Published figures on related setups are higher under
their own protocols, which is the point the paper is making rather than a ranking it
disputes. Whether a contrastive objective changes the feature result is untested.

## Citation

Proceedings entry and ACL Anthology link to follow.

## Contact

Ishan Katoch, ikatoch_be21@thapar.edu
