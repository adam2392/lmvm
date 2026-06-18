# LVMM Proof-of-Concept: Experiment Specification

## 1. System Overview

This experiment tests whether a vision model with a **fixed kernel-based Vision Core** and an **external Visual Knowledge Database** can (a) reason visually as well as a fully learned baseline, and (b) represent entity-specific visual knowledge in a modular, externalized way. The hypothesis, borrowing from LMLM, is that the model learns to issue lookups rather than memorizing entity appearances.

**Three systems are trained and compared:**

| System | Feature Extractor | Entity Knowledge | Training Signal |
|---|---|---|---|
| **LVMM** | Fixed RFF filter bank | External DB (injected at training time) | VQA CE loss |
| **LVMM-NoDB** | Fixed RFF filter bank | None (raw F_core) | VQA CE loss |
| **Baseline** | Fixed RFF filter bank (same) | Learned internally | VQA CE loss |

> **Note**: All three systems share the same fixed F_core backbone. The only difference is whether entity regions are replaced with DB prototypes (LVMM), kept raw (LVMM-NoDB), or kept raw but trained end-to-end with a learned head (Baseline). This isolates the effect of the database injection mechanism from any difference in feature quality.

---

## 2. Architecture

### 2.1 Vision Core: Data-Adaptive Random Fourier Features

The filter bank is built **once offline** from unlabeled training images. It is never updated.

**Construction algorithm:**

```
Input: N_patch = 200,000 random image patches sampled from training images
       Patch extraction: resize image to 224px, sample 16×16px crops uniformly
       Flatten each patch: p ∈ ℝ^768 (16×16×3)

Step 1 — Preprocessing:
   - Compute mean μ and std σ over all patches (per-channel)
   - Standardize: p ← (p - μ) / σ
   - Clip to [-3, 3]

Step 2 — PCA whitening (fit on the N_patch sample):
   - Fit PCA, retain K_pca = 128 components
   - Whitening transform: p_white = Λ^{-1/2} V^T (p - μ)
     where V = eigenvectors, Λ = eigenvalues
   - Save (μ, V, Λ) to disk

Step 3 — Random feature sampling (per scale):
   - For scale s ∈ {1, 2, 3}, independently sample:
     Ω_s ∈ ℝ^{128 × 256}: columns drawn from N(0, I_128)
     b_s ∈ ℝ^{256}: drawn from Uniform(0, 2π)
   - Save all (Ω_s, b_s) to disk

Step 4 — RFF map for a patch p:
   φ_s(p) = sqrt(2/256) · cos(Ω_s^T p_white + b_s) ∈ ℝ^{256}
```

**Application to a full image (multi-scale, produces F_core):**

```
Image I ∈ ℝ^{224×224×3}

Scale 1: extract 16px patches, stride 16px → 14×14 = 196 patches → apply φ_1 → ℝ^{14×14×256}
Scale 2: extract 32px patches, stride 32px → 7×7 → bilinear upsample to 14×14 → ℝ^{14×14×256}
Scale 3: extract 8px patches, stride 8px  → 28×28 → bilinear downsample to 14×14 → ℝ^{14×14×256}

Concatenate along channel dim:
F_core ∈ ℝ^{14×14×768}  (196 spatial tokens, 768-dim each)

F_core has NO learned parameters. It is deterministic given the sampled (Ω, b, V, Λ, μ).
```

**Implementation note:** Pre-compute and cache F_core for every training/validation image to an HDF5 file before training begins. Loading from cache is ~10× faster than recomputing per epoch.

### 2.2 Visual Knowledge Database

Stores one prototype vector per entity class, in the same 768-dim F_core feature space.

**Construction:**

```python
for each entity class e:
    crops = [bbox_crop(img, bbox) for (img, bbox, label) in annotations if label == e]
    features = [mean_pool_spatial(F_core(crop)) for crop in crops]  # each ∈ ℝ^768
    prototype[e] = mean(features)
    prototype[e] = prototype[e] / ||prototype[e]||  # L2 normalize

Build FAISS IndexFlatIP over all prototypes (inner product = cosine similarity after normalization)
```

**Retrieval:** Given a query region feature x ∈ ℝ^{768} (L2 normalized), return argmax_e cos(x, prototype[e]).

**Database is fixed during Reasoning Model training.** It can be updated (insert/delete) at any time without retraining.

### 2.3 Reasoning Model (shared architecture, LVMM and Baseline)

```
Input:  visual_tokens ∈ ℝ^{B × 196 × 768}   (already entity-injected for LVMM)
        question_tokens ∈ ℤ^{B × Q}           (tokenized question, Q ≤ 30)

Architecture:
  [CLS] token prepended → 197 total visual tokens
  Linear projection: 768 → d_model = 256
  Learned positional embeddings (197 positions)
  Question embedding: learned embedding table (vocab_size=3000) → ℝ^{Q × 256}
  Concatenate [visual_tokens || question_tokens] → ℝ^{(197+Q) × 256}
  6 × Transformer encoder block:
    MultiHeadAttention(d_model=256, n_heads=8, dropout=0.1)
    FFN(d_ff=1024, activation=GELU, dropout=0.1)
    LayerNorm (pre-norm)
  Extract [CLS] token output → ℝ^{256}
  Linear head → logits ∈ ℝ^{n_answers}
  Loss: cross-entropy

Total trainable params: ~4.2M
```

**Baseline difference:** In the Baseline, `visual_tokens` is the raw F_core output with no entity injection. LVMM-NoDB is identical. Only LVMM replaces entity-region tokens with DB prototypes before the Transformer.

### 2.4 Entity Injection Mechanism (LVMM only)

Called once per training step, before the forward pass through the Transformer.

```python
def inject_prototypes(F_core, bboxes, db, spatial_size=14):
    """
    F_core:  [B, 196, 768]  — spatial feature tokens
    bboxes:  List[List[BBox]] — ground-truth entity bboxes per image (normalized [0,1])
    db:      VisualKnowledgeDB
    Returns: F_injected: [B, 196, 768]
    """
    F_injected = F_core.clone()
    for b, img_bboxes in enumerate(bboxes):
        for bbox, entity_id in img_bboxes:
            # Convert normalized bbox to spatial token indices
            x0, y0, x1, y1 = bbox
            row_min = int(y0 * spatial_size)
            row_max = int(y1 * spatial_size) + 1
            col_min = int(x0 * spatial_size)
            col_max = int(x1 * spatial_size) + 1
            token_indices = [r * spatial_size + c
                             for r in range(row_min, row_max)
                             for c in range(col_min, col_max)]
            # Retrieve prototype from database (no grad)
            with torch.no_grad():
                query = F_core[b, token_indices].mean(0)
                prototype = db.retrieve(query)  # ℝ^768
            # Replace entity tokens with prototype
            F_injected[b, token_indices] = prototype
    return F_injected  # Transformer sees this; raw entity appearance never touches weights
```

**Key property:** The Transformer's weights are never trained on raw entity-region F_core features. During training they always see the DB prototype. At test time without the DB (LVMM-NoDB), the Transformer receives raw F_core for entity regions — something it has never been trained on.

### 2.5 Entity Classification Head (for memorization evaluation only)

A separate linear probe trained *after* main training, kept frozen from main training:

```
LVMM:     mean_pool(F_core[entity_region]) → Linear(768, n_entities)
Baseline: mean_pool(F_core[entity_region]) → Linear(768, n_entities)
```

This is fit on the training set with entity region crops and labels, providing a fair entity classification comparison that does not conflate VQA reasoning accuracy with entity memorization.

---

## 3. Datasets

### 3.1 Simulated: CLEVR

**Source:** http://cs.stanford.edu/people/jcjohns/clevr/ (CLEVR_v1.0.zip)

**Why CLEVR:** Ground-truth scene graphs eliminate annotation cost. Entity labels are exact (no noise). Reasoning question types are cleanly separable. Scales to a single GPU.

**Entity definition:**
- Each unique (color × shape × material × size) tuple = one entity type
- 8 colors × 3 shapes × 2 materials × 2 sizes = 96 possible, ~60-70 observed in practice
- Ground-truth entity identity: from scene JSON `color`, `shape`, `material`, `size` fields

**Bounding box construction** (CLEVR does not provide pixel-space bboxes directly):

```python
def clevr_bbox_from_pixel_coords(pixel_coords, shape, size):
    """
    pixel_coords: (x, y, depth) from scene JSON
    Returns approximate bbox as (x0, y0, x1, y1) in pixel space.
    """
    x, y = pixel_coords[0], pixel_coords[1]
    # Object radius in pixels: empirically ~30px for large, ~18px for small
    radius = 30 if size == 'large' else 18
    return (max(0, x - radius), max(0, y - radius),
            min(480, x + radius), min(320, y + radius))
```

> **Validation:** Visually inspect 50 random images with overlaid bboxes to confirm coverage before proceeding.

**Data splits:**

| Split | Images | Questions |
|---|---|---|
| Train | 70,000 | 699,960 |
| Val | 15,000 | 149,991 |
| Test | 15,000 | 149,988 |

**Answer vocabulary:** 28 classes (yes/no, 0-10 counts, colors, shapes, materials, sizes)

**Question types (from metadata, used in per-type breakdown):**

| Type | Count | Notes |
|---|---|---|
| `query_attribute` | ~33% | "What color is X?" — entity-dependent |
| `count` | ~20% | "How many X?" — spatial reasoning |
| `exist` | ~20% | "Is there a Y?" — spatial + entity |
| `compare_attribute` | ~15% | "Same size as?" — relational |
| `compare_integer` | ~12% | "More X or Y?" — counting + compare |

### 3.2 Real: GQA + Visual Genome

**GQA (questions):** https://cs.stanford.edu/people/dorarad/gqa/download.html
- Use: `train_balanced_questions.json` (943,000 questions), `val_balanced_questions.json` (132,062)
- 1,843 answer classes (balanced distribution)
- Images are a subset of Visual Genome (images must be downloaded separately)

**Visual Genome (entity bboxes):** https://visualgenome.org/api/v0/api_home.html
- Use: `objects.json` — provides bounding boxes + class labels for all objects
- Filter to objects with ≥ 30 training instances → ~500 entity classes
- Use VG image IDs to align with GQA questions (GQA is built on VG scene graphs)

**Image download:** VG images (part 1 + part 2, ~15GB total). GQA uses the same image set.

**Entity subset selection:**

```python
# From VG objects.json
entity_counts = Counter(obj['names'][0] for img in vg_objects for obj in img['objects'])
entity_classes = [name for name, count in entity_counts.items() if count >= 30]
# Result: ~400-500 entity classes
```

**GQA question type breakdown:**

| Structural Type | Semantic Type |
|---|---|
| verify (yes/no) | object, attribute, relation, category, global |
| query | same |
| choose (A or B) | same |
| logical (and/or) | same |
| compare | same |

Use GQA metadata field `"types"` to group questions in the evaluation.

---

## 4. Training Procedure

### Phase 0: Filter Bank Construction

```bash
python scripts/build_filter_bank.py \
    --image_dir data/clevr/images/train/ \
    --n_patches 200000 \
    --patch_size 16 \
    --n_pca_components 128 \
    --n_rff_per_scale 256 \
    --n_scales 3 \
    --output_dir checkpoints/filter_bank/
```

Outputs: `filter_bank.npz` containing `{mu, V, Lambda, Omega_1, b_1, Omega_2, b_2, Omega_3, b_3}`

**Then cache F_core for all training images:**

```bash
python scripts/cache_fcore.py \
    --image_dir data/clevr/images/train/ \
    --filter_bank checkpoints/filter_bank/filter_bank.npz \
    --output_file data/processed/clevr_train_fcore.h5 \
    --batch_size 128
```

Each entry in the HDF5 file is a (14, 14, 768) float16 array keyed by image filename.

### Phase 1: Database Construction

```bash
python scripts/build_database.py \
    --fcore_cache data/processed/clevr_train_fcore.h5 \
    --scene_json data/clevr/scenes/CLEVR_train_scenes.json \
    --output_dir checkpoints/database/
```

Outputs: `prototypes.npz` (entity_id → 768-dim float32 vector), `faiss_index.bin`, `entity_labels.json`

### Phase 2: Training

**LVMM:**

```bash
python train.py \
    --mode lvmm \
    --fcore_cache data/processed/clevr_train_fcore.h5 \
    --database checkpoints/database/ \
    --scene_json data/clevr/scenes/CLEVR_train_scenes.json \
    --questions data/clevr/questions/CLEVR_train_questions.json \
    --val_questions data/clevr/questions/CLEVR_val_questions.json \
    --d_model 256 --n_heads 8 --n_layers 6 --d_ff 1024 \
    --n_answers 28 --question_vocab_size 3000 --max_q_len 30 \
    --lr 1e-4 --weight_decay 1e-2 --batch_size 64 --epochs 30 \
    --warmup_frac 0.05 --grad_clip 1.0 \
    --output_dir checkpoints/lvmm/
```

**Baseline:**

```bash
python train.py \
    --mode baseline \   # Same script, different mode flag
    --fcore_cache data/processed/clevr_train_fcore.h5 \
    # No --database argument
    ...same hyperparameters...
    --output_dir checkpoints/baseline/
```

**LVMM-NoDB (ablation):** Use `--mode baseline` but load an LVMM checkpoint for eval.

---

## 5. Evaluation Protocol

### 5.1 Unit Test 1 — Filter Bank

Run before any model training. All tests use CLEVR validation images.

**Test FB-A: Color separability (primary correctness check)**
- Sample 100 crops per color class (8 colors) from validation scenes using ground-truth pixel_coords
- Extract F_core from each crop, mean-pool spatially → 768-dim feature
- Fit linear SVM (sklearn, 5-fold CV) on F_core features
- **Pass criterion:** accuracy ≥ 80%
- *If failed: check PCA whitening, confirm patch normalization, try increasing D.*

**Test FB-B: Shape separability**
- Same protocol, 3 shape classes (cube, sphere, cylinder)
- **Pass criterion:** accuracy ≥ 65%
- *If failed: increase patch sizes (shapes are coarser structure than colors).*

**Test FB-C: Entity type retrieval (coarser test)**
- Use 30 crops per entity type (color+shape+material combination, ~60 types)
- NN retrieval in F_core space: given a query crop, return the nearest gallery crop
- Ground truth positive: same entity type
- Metric: mAP@10
- **Pass criterion:** mAP@10 ≥ 0.20
- *Baseline: HOG features on same crops should achieve ~0.15-0.25 — F_core should be comparable.*

**Test FB-D: Sanity check — translation invariance on solid patches**
- Create a 224×224 single-color image, extract F_core at two different spatial positions
- Verify the feature vectors are identical (no position-dependent variation for uniform regions)
- **Pass criterion:** L2 distance < 1e-5

Report all four tests in a unit_test_filter_bank.json file before proceeding.

### 5.2 Unit Test 2 — Database

Run after Phase 1 construction and before Phase 2 training.

**Test DB-A: Prototype recall (primary correctness check)**
- Hold out 15% of entity exemplar crops per class as test queries
- Retrieve nearest prototype → entity class
- Metric: Top-1 and Top-5 accuracy over all query crops
- **Pass criterion (CLEVR):** Top-1 ≥ 80%, Top-5 ≥ 95%
- **Pass criterion (GQA):** Top-1 ≥ 50%, Top-5 ≥ 75%
- *If failed: likely too few exemplars per class; reduce entity set to classes with ≥ 50 examples.*

**Test DB-B: Prototype stability**
- Randomly split exemplars per entity class into halves A and B
- Compute prototype from each half
- Measure cosine similarity between prototype_A and prototype_B, averaged across classes
- **Pass criterion:** mean cosine similarity ≥ 0.80
- *If failed: F_core features may not be consistent enough; check for image augmentation in extraction.*

**Test DB-C: Few-shot registration**
- Identify 10 entity classes not used in database construction (hold out before Phase 1)
- Register each with K = 1, 5, 10, 20 exemplar crops
- Test with 30 query crops per class
- Report Top-1 accuracy vs K for each held-out class
- **Pass criterion:** accuracy increases monotonically with K; K=10 achieves ≥ 50%

**Test DB-D: Unlearning**
- Remove 5 entity classes from the FAISS index
- Query all 50 test crops for those 5 classes → verify accuracy drops to < 5%
- Query all test crops for the remaining classes → verify accuracy is unchanged (±2%)
- **Pass criterion:** both conditions met

Report all tests in unit_test_database.json.

### 5.3 Main Evaluation A — Visual Reasoning

**Protocol:**
- Evaluate on the full validation question set
- For LVMM: inject GT bboxes from scene/VG metadata (same as training)
- For Oracle-LVMM: inject the exact correct prototype (bypass retrieval, use entity label to look up)
- Report VQA accuracy (exact string match) overall and by question type

**CLEVR results table:**

```
System               | Overall | query_attr | count | exist | compare_attr | compare_int
---------------------|---------|------------|-------|-------|--------------|------------
Oracle-LVMM          | ?       | ?          | ?     | ?     | ?            | ?
LVMM                 | ?       | ?          | ?     | ?     | ?            | ?
LVMM-NoDB            | ?       | ?          | ?     | ?     | ?            | ?
Baseline             | ?       | ?          | ?     | ?     | ?            | ?
```

**Expected patterns:**
- Oracle-LVMM ≈ Baseline or better (upper bound on LVMM)
- LVMM ≈ Baseline on reasoning-heavy questions (count, compare_int)
- LVMM > Baseline on entity-attribute questions IF retrieval is accurate (query_attr)
- LVMM-NoDB < Baseline on entity-attribute questions (no entity knowledge)

**GQA results table (if running real-data experiment):**

```
System      | Overall | object | attribute | relation | category | global
------------|---------|--------|-----------|----------|----------|-------
Oracle-LVMM | ?       | ?      | ?         | ?        | ?        | ?
LVMM        | ?       | ?      | ?         | ?        | ?        | ?
LVMM-NoDB   | ?       | ?      | ?         | ?        | ?        | ?
Baseline    | ?       | ?      | ?         | ?        | ?        | ?
```

### 5.4 Main Evaluation B — Entity Memorization

**Task:** Given a bounding-box crop of an entity, classify its type.

**Classification method per system:**

| System | Method |
|---|---|
| LVMM+DB | Cosine NN to database prototypes (no model involved) |
| LVMM-NoDB | F_core of crop → mean pool → linear probe (fit after training) |
| Baseline | F_core of crop → mean pool → linear probe (fit after training) |
| F_core-only | F_core of crop → mean pool → cosine NN to prototypes (same as LVMM+DB) |

> Note: Both LVMM-NoDB and Baseline use the same F_core features for their linear probe. The question is whether the *Transformer's internal representations* differ (evaluated separately below).

**Test set:** 5,000 entity crops from validation images (balanced across entity classes)

**CLEVR entity classification results table:**

```
System       | Top-1 | Top-3
-------------|-------|------
LVMM+DB      | ?     | ?     ← should be high (retrieval-based)
LVMM-NoDB    | ?     | ?     ← should be lower (no entity knowledge)
Baseline     | ?     | ?     ← should be high (learned internally)
F_core-only  | ?     | ?     ← measures F_core's inherent discriminability
```

**Expected key result:** LVMM+DB ≈ Baseline (competitive entity recognition through different mechanisms), while LVMM-NoDB < Baseline (entity knowledge not in model weights).

**Transformer representation probe (deeper analysis):**
- Extract the [CLS] token embedding from the Transformer after processing an entity crop
- Fit a linear probe on these embeddings
- LVMM: should fail to classify entities from [CLS] (never trained on entity-region raw features)
- Baseline: should succeed (entity appearance encoded in weights)

```
System      | Probe from [CLS] embedding
------------|---------------------------
LVMM        | ≤ F_core-only accuracy  ← entity knowledge NOT in weights
Baseline    | >> F_core-only accuracy ← entity knowledge IS in weights
```

This directly tests the core claim.

**Unlearning curve:**
- Remove N = {0, 5, 10, 20, 30} entity classes from LVMM database
- Measure recognition accuracy on removed classes and retained classes after each deletion
- Plot: Accuracy vs. N deleted for {removed classes, retained classes}
- Baseline: same plot but requires retraining (just measure baseline accuracy unchanged)
- **Expected:** LVMM removed-class accuracy → 0 immediately; Baseline unchanged

**Few-shot addition curve:**
- Register K = {1, 3, 5, 10, 20} exemplars of 5 held-out entity classes into LVMM database
- Measure recognition accuracy on those classes vs K
- Baseline comparison: fine-tune Baseline model with same K crops (with early stopping)
- **Expected:** LVMM improves immediately with K; Baseline requires more data

---

## 6. Code Structure

```
lvmm_poc/
├── README.md
├── requirements.txt
├── configs/
│   ├── clevr_lvmm.yaml
│   ├── clevr_baseline.yaml
│   └── gqa_lvmm.yaml
├── scripts/
│   ├── build_filter_bank.py       # Phase 0
│   ├── cache_fcore.py             # Cache F_core to HDF5
│   ├── build_database.py          # Phase 1
│   ├── run_unit_tests.py          # Runs FB and DB unit tests, writes JSON reports
│   └── run_evaluation.py          # All main evaluations
├── lvmm/
│   ├── __init__.py
│   ├── filter_bank.py             # DataAdaptiveRFF class
│   ├── database.py                # VisualKnowledgeDB class
│   ├── injection.py               # inject_prototypes()
│   ├── model.py                   # ReasoningModel (LVMM and Baseline share this)
│   └── datasets/
│       ├── clevr.py               # CLEVRDataset
│       └── gqa.py                 # GQADataset
├── train.py                       # Main training script (mode flag: lvmm / baseline)
└── evaluate.py                    # Evaluation entry point
```

### 6.1 Key Classes

```python
# lvmm/filter_bank.py
class DataAdaptiveRFF:
    def fit(self, patches: np.ndarray) -> None
        # patches: [N, 768], fits PCA whitening + samples random directions
    def transform_image(self, image: np.ndarray) -> np.ndarray
        # image: [H, W, 3] uint8, returns [14, 14, 768] float32
    def transform_batch(self, images: torch.Tensor) -> torch.Tensor
        # images: [B, 3, H, W], returns [B, 196, 768]
    def save(self, path: str) -> None
    @classmethod
    def load(cls, path: str) -> 'DataAdaptiveRFF'

# lvmm/database.py
class VisualKnowledgeDB:
    def __init__(self, feature_dim: int = 768)
    def build(self, entity_features: Dict[str, List[np.ndarray]]) -> None
        # entity_features: {entity_id: [crop_feature_1, ...]}
    def retrieve(self, query: np.ndarray, k: int = 1) -> Tuple[np.ndarray, List[str]]
        # query: [768] float32, returns (prototype_vector, [entity_id])
    def register(self, entity_id: str, exemplars: List[np.ndarray]) -> None
        # Add or update an entity — O(1) operation
    def remove(self, entity_id: str) -> None
        # Remove entity — O(1) operation, rebuilds FAISS index
    def save(self, path: str) -> None
    @classmethod
    def load(cls, path: str) -> 'VisualKnowledgeDB'

# lvmm/model.py
class ReasoningModel(nn.Module):
    def __init__(self, input_dim=768, d_model=256, n_heads=8, n_layers=6,
                 d_ff=1024, n_answers=28, q_vocab_size=3000, max_q_len=30,
                 dropout=0.1)
    def forward(self, visual_tokens: Tensor, question_tokens: Tensor) -> Tensor
        # visual_tokens: [B, 196, 768]  (entity-injected or raw, depending on mode)
        # question_tokens: [B, Q] int
        # returns: [B, n_answers] logits
```

### 6.2 Dataset Contract

Both datasets must return the following from `__getitem__`:

```python
{
    "image_id": str,
    "fcore": torch.Tensor,           # [196, 768] — loaded from HDF5 cache
    "question_tokens": torch.Tensor,  # [Q] int
    "answer_idx": int,
    "entity_bboxes": List[Tuple[float, float, float, float]],  # [(x0,y0,x1,y1) normalized]
    "entity_ids": List[str],          # entity class label per bbox
}
```

CLEVR: bboxes derived from `pixel_coords` + shape/size heuristic (see §3.1). Entity ID = `f"{color}_{shape}_{material}_{size}"`.

GQA/VG: bboxes from VG `objects.json`, entity ID = object name.

---

## 7. Dependencies

```
# requirements.txt
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
scikit-learn>=1.2.0      # PCA, SVM for unit tests
faiss-gpu>=1.7.4         # (or faiss-cpu if no GPU available)
h5py>=3.8.0              # F_core cache
Pillow>=9.5.0
tqdm>=4.65.0
pyyaml>=6.0
wandb>=0.15.0            # optional logging
```

---

## 8. Compute Requirements

| Phase | Hardware | Est. Time |
|---|---|---|
| Filter bank fit (CLEVR) | CPU | 10-20 min |
| F_core caching — CLEVR (70k images) | 1× GPU (any) | 30-60 min |
| F_core caching — GQA (108k images) | 1× GPU | 1-2 hours |
| Database construction — CLEVR | 1× GPU | 10-20 min |
| Database construction — GQA | 1× GPU | 30-60 min |
| Training — CLEVR (30 epochs, bs=64) | 1× A100 or 2× V100 | 3-5 hours |
| Training — GQA (20 epochs, bs=64) | 2× A100 | 10-18 hours |
| All evaluations | 1× GPU | 1-2 hours |

**Full CLEVR PoC can run on a single A100 in ~6 hours end-to-end.**

---

## 9. Expected Results and Success Criteria

The PoC is considered **successful** if all of the following hold:

| # | Criterion | Metric | Target |
|---|---|---|---|
| 1 | Filter bank captures appearance | Color SVM (Test FB-A) | ≥ 80% |
| 2 | Database has discriminative prototypes | Top-1 entity recall (Test DB-A, CLEVR) | ≥ 80% |
| 3 | Unlearning is clean | Removed-entity accuracy (Test DB-D) | < 5% |
| 4 | Reasoning is competitive | LVMM vs Baseline overall VQA Δ | < 5 pp |
| 5 | Reasoning is separated from entity knowledge | LVMM-NoDB reasoning-type Q accuracy ≈ Baseline | Δ < 5 pp on count/compare |
| 6 | Entity knowledge externalized | LVMM-NoDB [CLS]-probe accuracy ≤ F_core-only accuracy | monotone ordering |
| 7 | Few-shot addition works | LVMM K=10 new-entity accuracy | ≥ 50% |

**Criterion 6 is the core theoretical claim**: the model's internal weights do not memorize entity appearances beyond what the fixed F_core already provides.

---

## 10. Execution Order for AI Agent

```
1. Download CLEVR dataset (images + questions + scenes)
2. Run: python scripts/build_filter_bank.py (CLEVR)
3. Run: python scripts/cache_fcore.py (CLEVR train + val)
4. Run: python scripts/build_database.py (CLEVR)
5. Run: python scripts/run_unit_tests.py --phase filter_bank
        → Read unit_test_filter_bank.json; abort if any test fails
6. Run: python scripts/run_unit_tests.py --phase database
        → Read unit_test_database.json; abort if any test fails
7. Run: python train.py --mode lvmm --config configs/clevr_lvmm.yaml
8. Run: python train.py --mode baseline --config configs/clevr_baseline.yaml
9. Run: python scripts/run_evaluation.py --reasoning --memorization
        → Produces results/clevr_results.json with all tables
10. (Optional) Repeat steps 1-9 for GQA dataset
```