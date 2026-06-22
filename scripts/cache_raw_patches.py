#!/usr/bin/env python
"""Cache raw RGB patch tokens to HDF5.

This is the no-filter-bank baseline cache.  It preserves the same downstream contract as
``cache_fcore.py``: one ``[14, 14, 768]`` float16 array per image filename.  Each token is
one non-overlapping 16x16 RGB patch from a 224x224 resized image, flattened in CHW order.
"""

import argparse
import glob
import os

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

IMAGE_EXTS = ("*.png", "*.jpg", "*.jpeg")
IMAGE_SIZE = 224
PATCH_SIZE = 16
GRID = IMAGE_SIZE // PATCH_SIZE
FEATURE_DIM = 3 * PATCH_SIZE * PATCH_SIZE


def list_images(image_dir):
    files = []
    for ext in IMAGE_EXTS:
        files.extend(glob.glob(os.path.join(image_dir, ext)))
    return sorted(files)


def load_batch(paths):
    imgs = []
    for path in paths:
        img = np.array(Image.open(path).convert("RGB"), copy=True)
        imgs.append(torch.from_numpy(img).permute(2, 0, 1))
    imgs = [F.interpolate(i.unsqueeze(0).float(), size=(IMAGE_SIZE, IMAGE_SIZE),
                          mode="bilinear", align_corners=False)[0] for i in imgs]
    batch = torch.stack(imgs, dim=0)
    if batch.max() > 1.5:
        batch = batch / 255.0
    return batch


def patchify(images):
    """[B,3,224,224] -> [B,14,14,768] raw patch tokens."""
    patches = images.unfold(2, PATCH_SIZE, PATCH_SIZE).unfold(3, PATCH_SIZE, PATCH_SIZE)
    patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
    return patches.view(images.shape[0], GRID, GRID, FEATURE_DIM)


@torch.no_grad()
def compute_stats(images, batch_size):
    total = 0
    sum_x = torch.zeros(FEATURE_DIM, dtype=torch.float64)
    sum_x2 = torch.zeros(FEATURE_DIM, dtype=torch.float64)
    for start in tqdm(range(0, len(images), batch_size), desc="stats"):
        batch = patchify(load_batch(images[start:start + batch_size])).view(-1, FEATURE_DIM)
        batch = batch.double()
        total += batch.shape[0]
        sum_x += batch.sum(0)
        sum_x2 += (batch * batch).sum(0)
    mean = sum_x / max(1, total)
    var = (sum_x2 / max(1, total)) - mean * mean
    std = torch.sqrt(torch.clamp(var, min=1e-12))
    return mean.float().numpy(), std.float().numpy()


def main():
    ap = argparse.ArgumentParser(description="Cache raw RGB patch tokens to HDF5.")
    ap.add_argument("--image_dir", required=True)
    ap.add_argument("--output_file", required=True)
    ap.add_argument("--stats_input", help="Existing .npz with mean/std for standardization.")
    ap.add_argument("--stats_output", help="Where to save computed train-set mean/std.")
    ap.add_argument("--no_standardize", action="store_true",
                    help="Store [0,1] raw patch values without train-set standardization.")
    ap.add_argument("--batch_size", type=int, default=1024)
    args = ap.parse_args()

    images = list_images(args.image_dir)
    if not images:
        raise SystemExit(f"No images found under {args.image_dir}")

    mean = std = None
    if not args.no_standardize:
        if args.stats_input:
            stats = np.load(args.stats_input)
            mean = stats["mean"].astype(np.float32)
            std = stats["std"].astype(np.float32)
        else:
            mean, std = compute_stats(images, args.batch_size)
            stats_out = args.stats_output or os.path.splitext(args.output_file)[0] + "_stats.npz"
            os.makedirs(os.path.dirname(os.path.abspath(stats_out)), exist_ok=True)
            np.savez(stats_out, mean=mean, std=std)
            print(f"Saved raw-patch standardization stats -> {stats_out}")
        mean = torch.from_numpy(mean).view(1, 1, 1, FEATURE_DIM)
        std = torch.from_numpy(std).view(1, 1, 1, FEATURE_DIM)

    print(f"Caching raw patches for {len(images)} images -> {args.output_file}")
    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
    with h5py.File(args.output_file, "w") as h5:
        for start in tqdm(range(0, len(images), args.batch_size), desc="caching"):
            batch_paths = images[start:start + args.batch_size]
            tokens = patchify(load_batch(batch_paths))
            if mean is not None:
                tokens = (tokens - mean) / (std + 1e-6)
            tokens = tokens.numpy().astype(np.float16)
            for path, arr in zip(batch_paths, tokens):
                key = os.path.basename(path)
                h5.create_dataset(key, data=arr, compression="gzip", compression_opts=4)
    print("Done.")


if __name__ == "__main__":
    main()
