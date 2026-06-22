# GCP A100 Runbook

This runbook assumes a Google Cloud GPU VM that syncs artifacts to/from GCS, but trains
from local disk.  Do not train directly from a mounted bucket; keep CLEVR, HDF5 caches,
and checkpoints on local SSD or fast persistent disk during the run.

## Machines

- 1 GPU: `a2-highgpu-1g` with 1x A100 40GB.
- 2 GPUs: `a2-highgpu-2g` with 2x A100 40GB.
- Use a PyTorch CUDA image or install CUDA-compatible PyTorch wheels through `uv`.

## Environment

```bash
uv venv .venv --python 3.11
uv pip install --python .venv/bin/python -r requirements.txt
```

## Data

```bash
mkdir -p data checkpoints results
curl -L -C - -o data/CLEVR_v1.0.zip https://dl.fbaipublicfiles.com/clevr/CLEVR_v1.0.zip
unzip -t data/CLEVR_v1.0.zip
unzip data/CLEVR_v1.0.zip -d data
mv data/CLEVR_v1.0 data/clevr
```

## Feature Caches

```bash
.venv/bin/python scripts/build_filter_bank.py \
  --image_dir data/clevr/images/train \
  --output_dir checkpoints/filter_bank

.venv/bin/python scripts/cache_fcore.py \
  --image_dir data/clevr/images/train \
  --filter_bank checkpoints/filter_bank/filter_bank.npz \
  --output_file data/processed/clevr_train_fcore.h5 \
  --batch_size 512

.venv/bin/python scripts/cache_fcore.py \
  --image_dir data/clevr/images/val \
  --filter_bank checkpoints/filter_bank/filter_bank.npz \
  --output_file data/processed/clevr_val_fcore.h5 \
  --batch_size 512

.venv/bin/python scripts/cache_raw_patches.py \
  --image_dir data/clevr/images/train \
  --output_file data/processed/clevr_train_rawpatch.h5 \
  --stats_output data/processed/rawpatch_stats.npz \
  --batch_size 2048

.venv/bin/python scripts/cache_raw_patches.py \
  --image_dir data/clevr/images/val \
  --output_file data/processed/clevr_val_rawpatch.h5 \
  --stats_input data/processed/rawpatch_stats.npz \
  --batch_size 2048

.venv/bin/python scripts/build_database.py \
  --fcore_cache data/processed/clevr_train_fcore.h5 \
  --scene_json data/clevr/scenes/CLEVR_train_scenes.json \
  --output_dir checkpoints/database \
  --n_holdout 10
```

## Unit Tests

```bash
.venv/bin/python scripts/run_unit_tests.py --phase filter_bank \
  --filter_bank checkpoints/filter_bank/filter_bank.npz \
  --image_dir data/clevr/images/val \
  --scene_json data/clevr/scenes/CLEVR_val_scenes.json

.venv/bin/python scripts/run_unit_tests.py --phase database \
  --database checkpoints/database
```

## One A100 Training

Start with these settings and increase `--batch_size` if `nvidia-smi dmon` shows memory
headroom and low utilization.

```bash
.venv/bin/python train.py --mode lvmm --config configs/clevr_lvmm.yaml \
  --batch_size 512 --num_workers 16 --amp bf16 --compile \
  --pin_memory --persistent_workers --prefetch_factor 4 --preload_cache

.venv/bin/python train.py --mode baseline --config configs/clevr_baseline.yaml \
  --batch_size 512 --num_workers 16 --amp bf16 --compile \
  --pin_memory --persistent_workers --prefetch_factor 4 --preload_cache

.venv/bin/python train.py --mode baseline --config configs/clevr_rawpatch_baseline.yaml \
  --batch_size 512 --num_workers 16 --amp bf16 --compile \
  --pin_memory --persistent_workers --prefetch_factor 4 --preload_cache
```

## Two A100 Training

`--batch_size` is per process/GPU.  The commands below use an effective batch size of
1024.  Keep the learning rate fixed for the first run, then tune if validation lags.

```bash
torchrun --nproc_per_node=2 train.py --mode lvmm --config configs/clevr_lvmm.yaml \
  --batch_size 512 --num_workers 12 --amp bf16 --compile \
  --pin_memory --persistent_workers --prefetch_factor 4 --preload_cache

torchrun --nproc_per_node=2 train.py --mode baseline --config configs/clevr_baseline.yaml \
  --batch_size 512 --num_workers 12 --amp bf16 --compile \
  --pin_memory --persistent_workers --prefetch_factor 4 --preload_cache

torchrun --nproc_per_node=2 train.py --mode baseline --config configs/clevr_rawpatch_baseline.yaml \
  --batch_size 512 --num_workers 12 --amp bf16 --compile \
  --pin_memory --persistent_workers --prefetch_factor 4 --preload_cache
```

## Evaluation

```bash
.venv/bin/python scripts/run_evaluation.py --reasoning --memorization \
  --lvmm_ckpt checkpoints/lvmm/best.pt \
  --baseline_ckpt checkpoints/baseline/best.pt \
  --rawpatch_ckpt checkpoints/rawpatch_baseline/best.pt \
  --lvmm_vocab_dir checkpoints/lvmm \
  --database checkpoints/database \
  --val_fcore_cache data/processed/clevr_val_fcore.h5 \
  --rawpatch_val_cache data/processed/clevr_val_rawpatch.h5 \
  --val_questions data/clevr/questions/CLEVR_val_questions.json \
  --val_scene_json data/clevr/scenes/CLEVR_val_scenes.json \
  --train_fcore_cache data/processed/clevr_train_fcore.h5 \
  --train_scene_json data/clevr/scenes/CLEVR_train_scenes.json \
  --filter_bank checkpoints/filter_bank/filter_bank.npz \
  --val_image_dir data/clevr/images/val
```

## Quick Smoke Test

```bash
.venv/bin/python train.py --mode baseline --config configs/clevr_rawpatch_baseline.yaml \
  --limit_train 2048 --limit_val 512 --epochs 1 --batch_size 128
```

## Monitoring

```bash
nvidia-smi dmon -s pucmt
```

If GPU utilization is low, prefer in this order: train from local disk, enable
`--preload_cache`, increase `--num_workers`, increase `--prefetch_factor`, then increase
batch size.  If memory is tight, drop `--preload_cache` before reducing batch size.
