# Related Work & Novelty Assessment

This document positions the LVMM proof-of-concept (see [`SPEC.md`](SPEC.md)) against prior
art. It is grounded in a fact-checked literature sweep (5 search angles, 23 primary sources,
25 adversarially-verified claims). The goal is an honest novelty assessment for the paper's
Related Work section — separating what is genuinely new from what is a recombination of
established techniques.

## TL;DR — novelty verdict

**No single prior paper combines all four of LVMM's defining elements**:

1. a fixed, non-end-to-end-trained random-Fourier **Vision Core**,
2. an external **editable prototype database**,
3. **train-time substitution** of entity-region tokens with retrieved prototypes (so the
   reasoner's weights never see raw entity appearance), and
4. a **memorization-externalization audit** (CLS-probe + clean per-class unlearning + few-shot
   addition).

LVMM is best characterized as a **novel recombination of well-established components**, not a
wholly new mechanism. Its single closest conceptual ancestors are **LMLM** (the borrowed
externalization idea, in language) and **MUNKEY** (the structurally closest vision analog).

## Closest prior art (the combination as a whole)

| System | What it shares with LVMM | How it differs |
|---|---|---|
| **LMLM** — Large Memory Language Models (Zhao et al., 2025) | The core goal LVMM borrows: externalize knowledge into an explicit, editable store so it isn't memorized in weights; editing/unlearning becomes a memory operation. | Language-only. **Loss-masks** retrieved factual *values* during training rather than substituting input prototypes. LVMM is effectively a vision analog. |
| **MUNKEY** (arXiv:2603.15033) | Structurally closest vision analog: **frozen (non-trained) encoder + external editable key-value memory + deletion-based zero-shot unlearning** (`M_u = M \ {(k_i,v_i) | i ∈ D_f}`). | Stores **learnable value tokens** (LVMM uses non-learned mean-pooled prototypes); **per-instance** granularity (LVMM is per-entity-class); **image classification**, not VQA. |

> ⚠️ **Caveat on MUNKEY.** It carries a future-dated 2026 arXiv id, is an unreviewed
> preprint (OpenReview `gGH3Xp1lHR`), and its headline figures (`~0s` unlearning, "beats nine
> baselines") are self-reported and not independently verified. Because it is the single
> closest structural precedent, the strength of the "closest prior art" claim is partly
> contingent on a non-peer-reviewed source. Verify its status before citing it as such.

## Component-by-component precedents

Every individual ingredient of LVMM has strong canonical precedent.

### (a) Externalizing knowledge so weights don't memorize it
- **LMLM** (Zhao et al., 2025) is the canonical and most-recent precedent and the explicit
  inspiration. No *published* vision analog was found other than MUNKEY (which was
  independently motivated by unlearning, not by LMLM).

### (b) Fixed / non-end-to-end-trained visual features (the `F_core` lineage)
- **Random Fourier Features** (Rahimi & Recht, NIPS 2008) — samples and **freezes**
  feature-map parameters `(w, b)`, fits only a linear readout, and uses exactly the
  `cos(wᵀx + b)` nonlinearity LVMM adopts (the `√(2/D)` is just variance normalization).
- **Scattering networks** (Bruna & Mallat, 2012) — cascade **fixed** wavelet filters with
  modulus + averaging; **no learning** in the features.
- **Random-weight CNNs** (Saxe et al., ICML 2011; Jarrett et al., 2009) — convolutional
  square-pooling architectures are provably frequency-selective and translation-invariant
  *even with random weights*; random filters reach ~53% on Caltech-101 vs. 54.2% for
  unsupervised-pretrained + finetuned, showing much of the performance comes from
  architecture, not learned weights.
- **PCANet** (Chan et al., 2015) — data-adaptive but *not* end-to-end: filters from PCA on
  patches, then fixed nonlinearities. **LVMM's core ≈ PCANet-style whitening + RFF, multi-scale.**

### (c) Prototype-based, neuro-symbolic, and retrieval-augmented VQA
- **ProtoVQA** (2025) — prototype-based VQA, but prototypes are **learned network parameters**,
  explicitly *not* externally managed via FAISS; no substitution or unlearning.
- **POEM** (CVPR 2023) — object-factorization prototypes inside an end-to-end neural module
  network; no external DB, substitution, or unlearning.
- **NS-VQA** (Yi et al., 2018) — disentangles perception from reasoning on CLEVR; the spirit of
  LVMM's "is entity knowledge in the weights?" probe, but no external editable memory.
- **REVEAL** (CVPR 2023) — external multimodal key-value memory with MIPS retrieval; **but
  trained end-to-end and recomputes embeddings during training**, so its "add/update without
  retraining" is an inference-time convenience, *not* a weights-don't-memorize externalization.
  A weaker analog than its surface description suggests.

### (d) Editable / unlearnable visual memory
- **MUNKEY** (2026) — unlearning as set-theoretic key deletion (see above).
- **Ma et al.** (CVPR 2018) — memory-augmented VQA, but end-to-end and motivated by long-tail
  sample-efficiency, *not* editability/externalization.
- An **ICLR 2025** result shows approximate unlearning methods fail to remove data-poisoning
  effects — which strengthens the motivation for *architectural* unlearning-by-design as in
  MUNKEY and LVMM.

## What is genuinely novel in LVMM

The novelty lives in the **specific synthesis**, not any single piece:

1. **Train-time substitution** of GT-bbox entity-region tokens with retrieved DB prototypes
   *before* the Transformer — so the reasoner is provably never trained on raw entity
   appearance. LMLM loss-masks *values*; MUNKEY *appends* an exemplar token. Neither replaces
   region appearance in the input the way LVMM does.
2. A **fully parameter-free path**: a fixed RFF extractor feeding **non-learned, mean-pooled,
   L2-normalized prototypes**. MUNKEY's values are learnable; ProtoVQA/POEM prototypes are
   learned.
3. The explicit **memorization-externalization audit in a VQA setting**: a linear probe on the
   `[CLS]` embedding showing entity knowledge is absent from the weights; per-class accuracy
   dropping to ~0 when a DB entry is deleted while others are unchanged; and a few-shot curve
   rising with K.

**Suggested positioning sentence for the paper:**
> *LVMM is a vision/VQA instantiation and extension of LMLM-style knowledge externalization,
> closest to MUNKEY in its unlearning-by-design, and distinguished by a fully non-learned
> feature-and-prototype path together with a perception-vs-memorization audit.*

## Caveats & open questions

- **Absence of evidence, not evidence of absence.** "No single paper combines all elements" is
  bounded by the searches performed; it is not a proof. A near-duplicate could exist among
  2025–2026 multimodal-memory preprints not surfaced here.
- **MUNKEY** is an unreviewed, future-dated preprint (see warning above).
- **REVEAL** is a weaker analog than its description implies (end-to-end, recomputed embeddings).
- **LVMM's own experimental claims** (CLEVR/GQA competitiveness, probe results, unlearning
  curves) are *stipulated from the design*, not yet produced on real data in this repo.
- Open question worth pre-empting in review: does the fixed RFF path or the reasoner's
  contextual mixing leak residual entity information after a DB deletion (analogous to MUNKEY's
  slightly-above-random membership-inference AUROC)? The unlearning audit (§5.4) should report
  this directly.

## References

**Knowledge externalization**
- LMLM — Pre-training Large Memory Language Models with Internal and External Knowledge — https://arxiv.org/abs/2505.15962

**Fixed / non-learned visual features**
- Random Features for Large-Scale Kernel Machines (Rahimi & Recht) — https://people.eecs.berkeley.edu/~brecht/papers/08.Rah.Rec.Allerton.pdf
- Invariant Scattering Convolution Networks (Bruna & Mallat) — https://arxiv.org/abs/1203.1513
- On Random Weights and Unsupervised Feature Learning (Saxe et al.) — http://robotics.stanford.edu/~ang/papers/nipsdlufl10-RandomWeights.pdf
- PCANet: A Simple Deep Learning Baseline for Image Classification (Chan et al.) — https://arxiv.org/abs/1404.3606

**Prototype / neuro-symbolic / retrieval VQA**
- ProtoVQA — https://arxiv.org/abs/2509.16680
- POEM (object factorization for compositional reasoning) — https://arxiv.org/pdf/2303.10482
- Neural-Symbolic VQA (NS-VQA) — https://arxiv.org/abs/1810.02338
- REVEAL: Retrieval-Augmented Visual-Language Pre-Training — https://openaccess.thecvf.com/content/CVPR2023/papers/Hu_REVEAL_Retrieval-Augmented_Visual-Language_Pre-Training_With_Multi-Source_Multimodal_Knowledge_Memory_CVPR_2023_paper.pdf
- Ma et al., Visual Question Answering with Memory-Augmented Networks (CVPR 2018) — https://openaccess.thecvf.com/content_cvpr_2018/papers/Ma_Visual_Question_Answering_CVPR_2018_paper.pdf

**Editable visual memory & machine unlearning**
- MUNKEY (memory-augmented transformer, unlearning by key deletion) — https://arxiv.org/html/2603.15033v3 · https://openreview.net/forum?id=gGH3Xp1lHR
- Approximate unlearning fails against data poisoning (ICLR 2025) — https://proceedings.iclr.cc/paper_files/paper/2025/file/7e810b2c75d69be186cadd2fe3febeab-Paper-Conference.pdf

---
*Generated from a fact-checked literature sweep on 2026-06-18 (25/25 claims verified, 0 refuted).
Treat preprint sources — especially MUNKEY — as provisional until peer-reviewed versions are confirmed.*
