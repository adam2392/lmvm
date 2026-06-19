#!/usr/bin/env python
"""Generate a tiny synthetic CLEVR-style dataset for integration testing.

Produces images with solid-colored shapes plus matching scene/question JSON in the exact
schema the CLEVR reader expects (pixel_coords, color/shape/material/size, program).
Color is strongly encoded (solid fills) so the color-SVM unit test is meaningful; this is
NOT a substitute for real CLEVR, only a pipeline smoke test.
"""

import argparse
import json
import os

import numpy as np
from PIL import Image, ImageDraw

COLORS = {
    "red": (220, 30, 30), "blue": (30, 30, 220), "green": (30, 180, 30),
    "yellow": (230, 220, 30), "purple": (150, 30, 200), "cyan": (30, 210, 210),
    "gray": (128, 128, 128), "brown": (140, 80, 30),
}
SHAPES = ["cube", "sphere", "cylinder"]
MATERIALS = ["rubber", "metal"]
SIZES = ["large", "small"]
W, H = 480, 320


def draw_object(draw, obj):
    x, y = obj["pixel_coords"][:2]
    r = 30 if obj["size"] == "large" else 18
    color = COLORS[obj["color"]]
    if obj["shape"] == "cube":
        draw.rectangle([x - r, y - r, x + r, y + r], fill=color)
    elif obj["shape"] == "sphere":
        draw.ellipse([x - r, y - r, x + r, y + r], fill=color)
    else:  # cylinder
        draw.ellipse([x - r // 2, y - r, x + r // 2, y + r], fill=color)


def make_program(fn):
    return [{"function": "scene"}, {"function": fn}]


def gen_split(out_dir, split, n_images, rng):
    img_dir = os.path.join(out_dir, "images", split)
    os.makedirs(img_dir, exist_ok=True)
    scenes, questions = [], []
    for i in range(n_images):
        fname = f"CLEVR_{split}_{i:06d}.png"
        n_obj = int(rng.integers(2, 5))
        img = Image.new("RGB", (W, H), (200, 200, 200))
        draw = ImageDraw.Draw(img)
        objs = []
        for _ in range(n_obj):
            obj = {
                "color": str(rng.choice(list(COLORS))),
                "shape": str(rng.choice(SHAPES)),
                "material": str(rng.choice(MATERIALS)),
                "size": str(rng.choice(SIZES)),
                "pixel_coords": [int(rng.integers(40, W - 40)),
                                 int(rng.integers(40, H - 40)),
                                 float(rng.uniform(5, 15))],
            }
            draw_object(draw, obj)
            objs.append(obj)
        img.save(os.path.join(img_dir, fname))
        scenes.append({"image_filename": fname, "objects": objs})

        # Questions: query_attribute (color), count, exist.
        target = objs[0]
        questions.append({"image_filename": fname,
                          "question": f"what color is the {target['shape']}",
                          "answer": target["color"], "program": make_program("query_color")})
        questions.append({"image_filename": fname, "question": "how many objects are there",
                          "answer": str(n_obj), "program": make_program("count")})
        present = str(rng.choice(list(COLORS)))
        exists = any(o["color"] == present for o in objs)
        questions.append({"image_filename": fname,
                          "question": f"is there a {present} thing",
                          "answer": "yes" if exists else "no",
                          "program": make_program("exist")})

    os.makedirs(os.path.join(out_dir, "scenes"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "questions"), exist_ok=True)
    with open(os.path.join(out_dir, "scenes", f"CLEVR_{split}_scenes.json"), "w") as f:
        json.dump({"scenes": scenes}, f)
    with open(os.path.join(out_dir, "questions", f"CLEVR_{split}_questions.json"), "w") as f:
        json.dump({"questions": questions}, f)
    print(f"{split}: {n_images} images, {len(questions)} questions")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n_train", type=int, default=80)
    ap.add_argument("--n_val", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    gen_split(args.out_dir, "train", args.n_train, rng)
    gen_split(args.out_dir, "val", args.n_val, rng)


if __name__ == "__main__":
    main()
