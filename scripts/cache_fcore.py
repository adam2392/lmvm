#!/usr/bin/env python
"""Cache F_core for every image to an HDF5 file (SPEC §2.1 / §4 Phase 0).

Each entry is a (14, 14, 768) float16 array keyed by image filename.  Loading from this
cache is ~10x faster than recomputing F_core per epoch.
"""

import argparse
import glob
import os
import sys

import h5py
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lvmm.filter_bank import DataAdaptiveRFF

IMAGE_EXTS = ("*.png", "*.jpg", "*.jpeg")


def list_images(image_dir):
    files = []
    for ext in IMAGE_EXTS:
        files.extend(glob.glob(os.path.join(image_dir, ext)))
    return sorted(files)


def load_batch(paths):
    imgs = []
    for p in paths:
        img = np.array(Image.open(p).convert("RGB"), copy=True)  # writable
        imgs.append(torch.from_numpy(img).permute(2, 0, 1))      # [3,H,W]
    # Resize handled inside transform_batch; stack requires equal size, so resize here.
    import torch.nn.functional as F
    imgs = [F.interpolate(i.unsqueeze(0).float(), size=(224, 224),
                          mode="bilinear", align_corners=False)[0] for i in imgs]
    return torch.stack(imgs, dim=0)


def main():
    ap = argparse.ArgumentParser(description="Cache F_core to HDF5.")
    ap.add_argument("--image_dir", required=True)
    ap.add_argument("--filter_bank", required=True)
    ap.add_argument("--output_file", required=True)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    fb = DataAdaptiveRFF.load(args.filter_bank).to(args.device)
    images = list_images(args.image_dir)
    if not images:
        raise SystemExit(f"No images found under {args.image_dir}")
    print(f"Caching F_core for {len(images)} images -> {args.output_file}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
    with h5py.File(args.output_file, "w") as h5:
        for start in tqdm(range(0, len(images), args.batch_size), desc="caching"):
            batch_paths = images[start : start + args.batch_size]
            batch = load_batch(batch_paths).to(args.device)
            fcore = fb.transform_batch(batch)                       # [B,196,768]
            fcore = fcore.view(-1, 14, 14, 768).cpu().numpy().astype(np.float16)
            for path, arr in zip(batch_paths, fcore):
                key = os.path.basename(path)
                h5.create_dataset(key, data=arr, compression="gzip", compression_opts=4)
    print("Done.")


if __name__ == "__main__":
    main()
