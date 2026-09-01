# Research Gaps: Recurring Patterns Across the Six Papers (Steps 12, 13, 15)

Tags: **[PAPER]** claimed · **[CODE]** verified in source · **[LOG]** shipped logs · **[OBS]** measured by me · **[INF]** inference

---

## Part A — What repeatedly *works*

| Technique | Papers | Evidence | Why it works |
|---|---|---|---|
| **Project *features*, not pixels or detections** | P1, P2, P3, P6 | **[PAPER]** P1 Table 3: pixels 26.8 → detections 68.2 → **features 88.2** MODA | A deep feature vector already summarises its receptive field, so it survives the spatial-structure break that warping causes |
| **Large receptive field *after* fusion** | P1, P2, P4 | P1 large-kernel **+11.3 MODA**; P2 BEV-FPN **+5.3 MODA**; P4 FRM (7×7) +0.4 MODA / +0.7 MODP | Fused BEV/common-space features are locally ambiguous; neighbourhood context resolves them. Replaces CRF/mean-field with plain convolution |
| **Permutation-invariant, N-agnostic pooling** | P2 (max), P3 (mean), P5 (softmax), P6 (sum) | ⭐ **[CODE]** P3 trains with `n_images=20` and tests with `50` — impossible for P1/P4 | The architecture survives a changed camera rig. P2's largest margin is on GMVD (**+4.5 MODA**), the only dataset where N varies across scenes |
| **Auxiliary supervision from a modality dropped at inference** | P3 (depth→occupancy), P6 (LiDAR+masks→depth/foreground) | P3 geometry shaping **+6.1/+4.6 mAP** at +26 ms; P6 In-Box **+2.2 mAP / 0 params** | Injects geometric priors during training without a deployment-time sensor cost |
| **Pretrained 2D backbones** | all six | P6 uses **nuImages**-pretrained, not ImageNet | Multi-view data is small; 2D pretraining carries most of the semantics |
| **Explicit cost reporting + parameter-matched controls** | P1, P2, P5, P6 | P5: 5 M params @ 35 FPS matches FIERY's 7 M @ 8 FPS with the *same backbone and decoder*; P2 R18 (22.9 M, 80.2) beats baseline R101 (43.2 M, 75.8) | Rules out "it's just bigger" — the standard reviewers will apply to you |
| **Shipping matched ablation configs** | P1 (`--variant`), P3 (`imvoxelnet_*.py`), P5 (`no_image_features`) | Verified by diffing P3's two configs: they differ **only** by `occ_head` + depth loading | Makes ablations reproducible as config diffs, not code branches |

---

## Part B — What repeatedly *fails* — the recurring gaps

### ⭐ Gap 1 — **Cross-view fusion is a fixed, unlearned reduction in 5 of 6 papers**

| Paper | Fusion | Learned? | Location |
|---|---|---|---|
| P1 MVDet | channel concat | ✗ | **[CODE]** `persp_trans_detector.py:77` |
| P2 MSMVD | max over views | ✗ | **[PAPER]** §3.1 |
| P3 ImGeoNet | mean (÷valid_count) | ✗ | **[CODE]** `imgeonet.py:103-107` |
| P4 CaMuViD | channel concat | ✗ | **[CODE]** `Custom_TwoStageDetector.py:716` |
| **P5 CVT** | **softmax attention** | ⭐ **✓** | **[CODE]** `encoder.py:156-161` |
| P6 GeoBEV | sum | ✗ | **[CODE]** `rcsample.py:378` |

Six papers, 2020–2025, and **exactly one** contributes anything to cross-view fusion itself. The other five innovate on *lifting* (P1, P2, P3, P6) or *supervision* (P3, P6) and reuse a parameter-free reduction unchanged from earlier work.

**Why this matters — [OBS] the data says views are wildly unequal:**
- Wildtrack camera coverage ranges from **C6 96.3 %** to **C4 27.7 %** of pedestrian instances (3.5×).
- Yet MVDet gives C4 exactly 512 of 3586 fusion channels, always; GeoBEV adds its contribution with weight 1.0, always.

> **Gap statement.** *All but one method aggregate views with a fixed, content-independent reduction, despite ground-truth evidence that views differ by more than 3× in how many objects they observe and by a median 2.7× in the scale at which they observe them.*

---

### ⭐ Gap 2 — **"More views is better" is assumed, and is demonstrably false**

**[PAPER]** P4 Table 3, cross-checked against **[OBS]** E2:

| added camera | Δ oracle coverage | Δ recall | Δ MODA |
|---|---|---|---|
| C4 | **+0.00 pp** | +3.4 | +3.2 |
| C5 | **+0.00 pp** | +1.4 | +0.0 |
| **C7** | **+0.40 pp** | **−0.7** | **−0.6** |

**The 7th camera reduces MODA from 95.6 to 95.0.** CaMuViD's own best result uses **six** cameras, not seven; the headline number is the worse one.

Corroborating evidence for saturation elsewhere:
- **[PAPER]** P3 Table 4: ARKit mAP@0.25 rises **+14.1** for views 10→20 but only **+2.2** for 50→100. Marginal value collapses ~20×.
- **[OBS]** E1: on Wildtrack, **three** cameras (C6+C3+C1) cover 99.5 % of pedestrians; on MultiviewX **two** (C3+C4) cover 100 %.

> **Gap statement.** *No method estimates the marginal informativeness of a view. Redundant views therefore consume linear compute for zero coverage gain, and can reduce accuracy by injecting duplicate or spurious evidence that uniform fusion cannot down-weight.*

---

### ⭐ Gap 3 — **Coverage and disambiguation are conflated**

Two distinct reasons an extra view helps:
1. **Coverage** — it sees an object no other view sees.
2. **Disambiguation** — it corroborates/refutes evidence about an object others already see.

**[OBS]** E2 separates them for the first time: cameras C4 and C5 provide **zero** coverage gain yet **+4.8 pp** of recall — pure disambiguation.

Every fusion operator in the six papers treats both identically. Worse, three of them *destroy* the disambiguation signal:
- **max** (P2) keeps only the strongest view — agreement between two views is indistinguishable from one strong view.
- **softmax** (P5) normalises weights to sum to 1 — six agreeing views look like one confident view.
- **sum** (P6) confounds "many weak views" with "one strong view".

Only **P3** computes cross-view **variance** — and then **[CODE]** `imgeonet.py:142,167` feeds it *only* to the occupancy head, never to the detector.

> **Gap statement.** *The only signal that distinguishes corroboration from coverage — cross-view agreement — is either never computed (P1, P2, P4, P5, P6) or computed and discarded before recognition (P3).*

---

### Gap 4 — **Missing / degraded views are barely studied, and the one study conflates two things**

| Paper | Missing-view study | Method |
|---|---|---|
| P1, P2, P3, P6 | ❌ none | — |
| P5 | ✅ dropout 0–3 cameras | Genuinely removes views from an N-agnostic model |
| P4 | ⚠ "camera elimination" | **[CODE]** feeds a **black image**; model not retrained; blanked cameras' GT still scored |

Feeding a black image is an **out-of-distribution corrupted input**, not an absent one. **[INF] Three regimes must be distinguished and never are:** (a) blank the view, (b) mask it out of an N-agnostic pooling, (c) retrain at that N.

> **Gap statement.** *Robustness to incomplete multi-view observation is essentially unmeasured, and the single fixed-N study that exists measures input corruption rather than view absence.*

---

### Gap 5 — **View-count curves are compared across rigs of incomparable redundancy**

**[INF]** The most important cross-paper insight, stated by none of them:

| Rig | Redundancy | Dropout behaviour |
|---|---|---|
| nuScenes surround (P5, P6) | **Low** — cameras tile 360°, marginal overlap | **[PAPER]** P5 Fig. 4: **linear** decay |
| Wildtrack (P1, P2, P4) | **Very high** — **[OBS]** mean 5.26/7 cameras per person, C1↔C6 Jaccard 92.8 % | **[PAPER]** P4 Table 3: steep, then flat, then **negative** |
| ScanNet/ARKit (P3) | High, sequential | **[PAPER]** Table 4: smooth **saturation** |

> **Gap statement.** *The shape of the accuracy-vs-view-count curve is dominated by the geometric redundancy of the rig, not by the fusion method — so no published claim of the form "our fusion scales better with views" is currently falsifiable. No paper reports a rig-redundancy statistic.*

**Proposed remedy (cheap, adoptable immediately):** report **mean views per object** and **pairwise view Jaccard** alongside every view-count curve. Both are computable from annotations alone — see `09_scripts/view_coverage.py`.

---

### Gap 6 — **Multi-scale is handled *within* views but not *across* them**

**[OBS]** E3: on Wildtrack, the same pedestrian spans a **median 2.73×** height ratio across cameras; **56.5 %** of pedestrians would be assigned to **different FPN levels** in different views (MultiviewX: 30.1 %).

| Paper | Multi-scale image features | Multi-scale *fused* representation |
|---|---|---|
| P1 | ✗ (single dilated level) | ✗ |
| P2 | ✅ 3 levels | ⭐ **✅ multi-scale BEV + BEV-FPN** — the only one |
| P3 | ✅ FPN, but fused **before** lifting | ✗ (single voxel resolution) |
| P4 | ✅ FPN levels | ✗ — each level fused **independently**, never interacting |
| P5 | ✅ 2 scales, iterative refinement | ⚠ partial |
| P6 | ✅ `scale_num=2` **within a view** | ✗ (single BEV resolution) |

> **Gap statement.** *Only MSMVD keeps scale information alive through view aggregation — and MSMVD has no public implementation.*

---

### Gap 7 — **Fixed-N architectures are still being published in 2025**

**[CODE]** P1: `nn.Conv2d(out_channel * self.num_cam + 2, 512, 3)`. **[CODE]** P4: `nn.Conv2d(256*self.num_head, 256, 1)` + N replicated RPN/ROI/refining heads. **[CODE]** P4 `config.py:12-13` requires manually setting `num_head`/`num_views` per dataset.

Consequences: cannot evaluate with fewer views without retraining; not permutation-invariant; parameter count grows linearly in N (P1's fusion head is **~19 M at N=7**, 1.7× its backbone).

> **Gap statement.** *Two of six papers — including one from 2025 — bake the camera count into the layer shapes, which forecloses every view-count, view-selection and view-order experiment.*

---

### Gap 8 — **Evaluation protocols have fragmented, and results are tabulated as if they had not**

- **[CODE]** P4 `evaluation.py:43-116` computes TPs by **2D image-space box IoU ≥ 0.45** with per-identity matching, while every competitor in its Table 1 uses **BEV Euclidean distance ≤ 0.5 m**. The paper discloses this (§4.2) but frames it around MODP only.
- **[CODE]** P1 `evaluate.py:21-33` has a **bare `except:`** that silently swaps the official MATLAB devkit for an unofficial Python re-implementation that the repo's own README says is *"approximately 0~2 % lower in MODA, MODP"*.
- **[CODE]** P5 `metrics.py:65-71` masks out low-visibility cells from the reported IoU — comparisons require the same `min_visibility`.

> **Gap statement.** *Headline numbers across this literature are not always measuring the same thing, and one of the two discrepancies is silent.*

---

### Gap 9 — **Recognition is thin: four of six papers predict a single class**

| Paper | Categories |
|---|---|
| P1, P2, P4 | 1 (pedestrian) |
| P5 | 2 binary masks (vehicle, drivable) — no instances |
| **P3** | ⭐ **18 / 189 / 17** |
| **P6** | ⭐ **10** |

**[LOG]** And where categories *are* predicted, the long tail collapses: **63 of 189 ScanNet200 classes (33.3 %) have AP@0.25 = 0 and AR@0.25 = 0** — never recalled once.

> **Gap statement.** *For a literature called "multi-view **recognition**", most of it is single-class localisation. The one paper that genuinely tests many categories fails on a third of them.*

---

### Gap 10 — **Reproducibility is poor across the board**

**[OBS]** verified this session across all six folders:

| Deficiency | Papers affected |
|---|---|
| **No pretrained weights present** | **all six** (0 `.pth` files found) |
| **Weights never released at all** | P3, P5 |
| **No code at all** | P2 |
| No hardware/runtime reported | P2 (none), P4 (none), P6 (no memory/training time) |
| Requires ≥2 GPUs | P1 (2, hard-coded), P6 (8) |
| Custom CUDA op needing `nvcc` | P3, P4, P6 |
| Deprecated pinned stack | P3 (torch 1.7.1/mmcv 1.2.7), P4 (mmcv 1.5.0), P5 (torchmetrics 0.7.2 — `compute_on_step` removed in ≥0.11), P6 (mmcv 1.5.3) |
| Published number from an **unconverged** run | **[LOG]** P3 ScanNet200: epoch **11 of 30**, second LR drop never fired |
| Config ≠ paper protocol | P5 (`max_steps: 30001` vs "30 epochs"; `dim_head` 32 vs 64), P6 (no 24-epoch/no-CBGS ablation config shipped) |
| Undocumented stray configs | P6 (`configs/my/`, 11 files) |

> **Gap statement.** *Not one of the six works can be reproduced end-to-end from what is on disk, and two of the five with code never released weights at all.*

---

## Part C — Gap intersection: where the opportunity is

Placing the three most transferable mechanisms against the three axes that matter:

| | Learned view weighting | Cross-view agreement / count | Multi-scale through fusion |
|---|:-:|:-:|:-:|
| P1 MVDet | ✗ | ✗ | ✗ |
| P2 MSMVD | ✗ | ✗ | ⭐ ✓ |
| P3 ImGeoNet | ✗ | ⚠ computed, **discarded before recognition** | ✗ |
| P4 CaMuViD | ✗ | ✗ | ✗ |
| P5 CVT | ⭐ ✓ | ✗ (softmax normalises it away) | ⚠ partial |
| P6 GeoBEV | ✗ | ✗ | ✗ (within-view only) |
| **Target** | ⭐ **✓** | ⭐ **✓** | ⭐ **✓** |

**[INF] No published method in this set occupies more than one cell.** The empty intersection is the defensible position for your research:

> **A multi-view aggregation operator that is permutation-invariant and N-agnostic, that predicts a per-(location, view) informativeness weight from content and geometry, that preserves cross-view agreement statistics (count and variance) rather than normalising them away, and that operates scale-by-scale.**

Each of its three components is individually validated by evidence in this paper set (P5's +5.0 IoU from camera-aware attention; P3's variance already computed for free; P2's +3.1 MODA from multi-scale projection), but no one has combined them — and, per Gap 2, none of them has been evaluated against the failure mode that matters most: **adding a view must never make things worse.**
