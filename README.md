# LVMM Proof-of-Concept

Implementation of the LVMM (Latent Visual Memory Model) experiment specified in
[`SPEC.md`](SPEC.md).  It tests whether a vision model with a **fixed kernel-based Vision
Core** plus an **external Visual Knowledge Database** can reason as well as a fully-learned
baseline while keeping entity-specific visual knowledge modular and externalized.

Three systems share the identical fixed `F_core` backbone and Transformer architecture and
differ only in how entity regions are handled:

| System        | Entity regions at train time         | At test time            |
|---------------|--------------------------------------|-------------------------|
| **LVMM**      | replaced with DB prototypes (inject) | retrieval / oracle / no-DB |
| **LVMM-NoDB** | (same weights as LVMM)               | raw `F_core` (no DB)    |
| **Baseline**  | raw `F_core`, learned end-to-end     | raw `F_core`            |

## Layout

```
lvmm/
  filter_bank.py   DataAdaptiveRFF — fixed multi-scale RFF Vision Core (§2.1)
  database.py      VisualKnowledgeDB — FAISS prototype store (§2.2)
  injection.py     inject_prototypes / inject_oracle (§2.4)
  model.py         ReasoningModel — shared Transformer (§2.3)
  vocab.py         question / answer vocabularies
  runtime.py       shared train/eval helpers (model build, LR schedule, injection)
  datasets/        clevr.py, gqa.py, common.py (dataset contract §6.2)
scripts/
  build_filter_bank.py   Phase 0 — fit the filter bank
  cache_fcore.py         cache F_core -> HDF5 (float16, keyed by filename)
  build_database.py      Phase 1 — build prototypes + FAISS index
  run_unit_tests.py      FB-A..D (§5.1) and DB-A..D (§5.2) -> JSON reports
  run_evaluation.py      reasoning (§5.3) + memorization (§5.4) tables
train.py                 Phase 2 — train (--mode lvmm | baseline)
evaluate.py              single-system VQA accuracy
tests/                   synthetic dataset generator + integration smoke test
```

## Setup

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt          # use faiss-gpu instead of faiss-cpu if you have a GPU
```

> **macOS note:** if PyTorch and FAISS are both linked against OpenMP you may see
> `OMP: Error #15`.  Export `KMP_DUPLICATE_LIB_OK=TRUE` before running (a known
> torch+faiss-on-macOS interaction, unrelated to this code).

## Execution order (CLEVR, SPEC §10)

```bash
# 1. Download CLEVR_v1.0 to data/clevr/{images,scenes,questions}

# 2. Phase 0 — filter bank + F_core cache
python scripts/build_filter_bank.py --image_dir data/clevr/images/train --output_dir checkpoints/filter_bank
python scripts/cache_fcore.py --image_dir data/clevr/images/train --filter_bank checkpoints/filter_bank/filter_bank.npz --output_file data/processed/clevr_train_fcore.h5
python scripts/cache_fcore.py --image_dir data/clevr/images/val   --filter_bank checkpoints/filter_bank/filter_bank.npz --output_file data/processed/clevr_val_fcore.h5

# 3. Phase 1 — database (use --n_holdout 10 to enable the few-shot test DB-C)
python scripts/build_database.py --fcore_cache data/processed/clevr_train_fcore.h5 \
    --scene_json data/clevr/scenes/CLEVR_train_scenes.json --output_dir checkpoints/database --n_holdout 10

# 4. Unit tests (each aborts with non-zero exit if a pass criterion fails)
python scripts/run_unit_tests.py --phase filter_bank \
    --filter_bank checkpoints/filter_bank/filter_bank.npz \
    --image_dir data/clevr/images/val --scene_json data/clevr/scenes/CLEVR_val_scenes.json
python scripts/run_unit_tests.py --phase database --database checkpoints/database

# 5. Phase 2 — train both systems
python train.py --mode lvmm     --config configs/clevr_lvmm.yaml
python train.py --mode baseline --config configs/clevr_baseline.yaml

# 6. Evaluation -> results/clevr_results.json
python scripts/run_evaluation.py --reasoning --memorization \
    --lvmm_ckpt checkpoints/lvmm/best.pt --baseline_ckpt checkpoints/baseline/best.pt \
    --lvmm_vocab_dir checkpoints/lvmm --database checkpoints/database \
    --val_fcore_cache data/processed/clevr_val_fcore.h5 \
    --val_questions data/clevr/questions/CLEVR_val_questions.json \
    --val_scene_json data/clevr/scenes/CLEVR_val_scenes.json \
    --train_fcore_cache data/processed/clevr_train_fcore.h5 \
    --train_scene_json data/clevr/scenes/CLEVR_train_scenes.json \
    --filter_bank checkpoints/filter_bank/filter_bank.npz \
    --val_image_dir data/clevr/images/val
```

## Quick integration test (no dataset download)

Generates a tiny synthetic CLEVR-style dataset and runs the whole pipeline:

```bash
KMP_DUPLICATE_LIB_OK=TRUE python tests/gen_synthetic_clevr.py --out_dir /tmp/synth/clevr
# then run the scripts above pointing at /tmp/synth/...  (see tests/ for a worked example)
```

On the synthetic data the filter-bank unit tests pass (color SVM ≈0.99, shape ≈0.92,
mAP@10 ≈0.48, translation-invariance L2 ≈2e-8) and DB-D unlearning is exact
(removed-class accuracy → 0.0, retained unchanged).

## Implementation notes / deviations from the spec

- **Cross-scale patches.** Patches at scales 2/3 (32px / 8px) are bilinearly resized to
  16×16 before flattening, so the single PCA-whitening basis (fit on 16px patches) applies
  to every scale; only the random projection `(Ω_s, b_s)` differs per scale (SPEC §2.1).
- **Database features.** Prototypes are built by mean-pooling the *cached* `F_core` tokens
  inside each labelled bbox — exactly the feature used as the injection query — so the
  prototype and query spaces coincide by construction.  This matches the
  `build_database.py` CLI contract (`--fcore_cache` + `--scene_json`, no image dir).
- **Parameter count.** The architecture matches §2.3 exactly; the resulting model is
  ~5.0M trainable params (the question embedding table dominates the difference from the
  spec's "~4.2M" estimate).
- **GQA/VG** readers (`lvmm/datasets/gqa.py`, `--dataset gqa` in the scripts) follow §3.2
  but are untested against real GQA data here.
