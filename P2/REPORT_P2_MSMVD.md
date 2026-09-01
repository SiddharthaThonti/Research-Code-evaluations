# P2 — MSMVD: Exploiting Multi-scale Image Features via Multi-scale BEV Features for Multi-view Pedestrian Detection

**Analysis date:** 2026-08-29
**Evidence base:** paper PDF only (`P_2_2508.20447v1_M.pdf`, 10 pp. + refs)
**Execution status:** ❌ **no code repository exists in this folder.** Nothing can be run, traced, or verified against source.

Tags: **[PAPER]** claimed · **[CODE]** verified in source · **[OBS]** measured by me · **[INF]** my inference

---

## 1. Inventory

| Item | Value |
|---|---|
| Title | MSMVD: Exploiting Multi-scale Image Features via Multi-scale BEV Features for Multi-view Pedestrian Detection |
| Authors | Taiga Yamane, Satoshi Suzuki, Ryo Masumura, Shota Orihashi, Tomohiro Tanaka, Mana Ihori, Naoki Makishima, Naotaka Kawata |
| Affiliation | NTT Human Informatics Laboratories, NTT Corporation, Yokosuka, Japan |
| Venue / Year | arXiv **2508.20447v1** (Aug 2025). Formatting (running head `YAMANE ET AL.: MSMVD`, "© 2025. The copyright of this document resides with its authors") is the **BMVC** template — **[INF]** likely a BMVC 2025 submission/paper; treat the venue as *unconfirmed*. |
| **Code repository** | ❌ **ABSENT.** The folder contains only the PDF. The paper states no code URL. |
| README / configs / scripts / weights | ❌ none |
| Result files | ❌ none |
| Reproducibility | **Impossible.** Not "blocked by resources" — there is nothing to run. |

> ⚠ **This is the only one of your six papers with no implementation.** Every claim below is a *paper claim*. It cannot be checked against source, and its numbers cannot be reproduced by anyone without a full re-implementation.

---

## 2. Relevance to *Multi-View Object Recognition*

**A. Problem solved.** Multi-View Pedestrian Detection (MVPD): predict a bird's-eye-view (BEV) occupancy map of pedestrians from *N* calibrated overlapping camera views. Same task as P1 (MVDet).

**B. What "multi-view" means here.** **Multiple fixed calibrated cameras** with partially overlapping FoV observing one shared ground region. Not multi-viewpoint object capture, not active/sequential views.

**C. Input.** `{Iⁿ}ⁿ₌₁ᴺ`, `Iⁿ ∈ ℝ^{3×H×W}`, plus per-camera intrinsics `Kⁿ ∈ ℝ^{3×3}` and extrinsics `[Rⁿ|Tⁿ] ∈ ℝ^{3×4}`. Images resized to **720 × 1280**.

**D. Output.** **Multi-scale BEV occupancy maps** `{M₃, M₄, M₅}` at three resolutions, plus offset maps `{O₃, O₄, O₅}`, merged at inference into one map `M ∈ ℝ^{1×X/2×Y/2}`. Single class (pedestrian); no category output.

**E. Relevance score.**

| Reading | Score | Why |
|---|---|---|
| *Multi-camera scene recognition* | **3.5 / 5** | Same task family; its **core idea (scale-aware view fusion) is the single most transferable concept in your whole set** |
| *Multi-viewpoint object recognition* | **2 / 5** | Task differs; but the observation that *the same object appears at wildly different scales in different views* is universal to multi-view recognition |

**Net: 3 / 5 conceptually, 1 / 5 practically** (no code). **[INF]** Treat P2 as an **idea source**, not a baseline. Its absence of code is, for you, an *opportunity*: the idea is published but not available, so a careful re-implementation is legitimate research work.

---

## 3. Research problem

- **Input:** N calibrated RGB views of a shared ground plane.
- **Output:** BEV pedestrian occupancy.
- **Assumptions:** synchronised cameras; known calibration; pedestrians on the ground plane; **fixed N per dataset** (though GMVD varies N *across scenes*).
- **Objective:** maximise MODA.

---

## 4. Motivation and the exact gap **[PAPER]**

Existing end-to-end MVPD methods (Fig. 1a) extract image features from **one** layer of the encoder (typically the last conv layer), project that **single-scale** feature to BEV, and predict **one** occupancy map.

The paper identifies two failure modes this causes (Fig. 2, vs MVFP):
1. Pedestrians that are **consistently small or consistently large** across all views are poorly represented by one fixed feature resolution.
2. Pedestrians whose **scale differs vastly between views** — the "important scale" for detecting them changes from camera to camera, so a single shared level cannot serve all views.

Monocular detection solved (1) long ago with FPN/PAFPN. **[PAPER]** *"Despite their benefits, multi-scale image features have not been used in end-to-end MVPD methods."*

### ✅ I verified this motivation empirically — **[OBS]**

Because P2 uses the same Wildtrack/MultiviewX benchmarks, I measured the cross-view scale variation directly from the ground-truth annotations shipped in `P4/CaMuViD/data` (script: `research_analysis/09_scripts/scale_variation.py`; output: `research_analysis/04_results/E3_*`):

| Statistic (same person, across the cameras that see them) | Wildtrack test | MultiviewX test |
|---|---|---|
| Median box-height ratio max/min | **2.73×** | **2.89×** |
| p90 / p95 | 4.95× / 5.59× | 6.13× / 6.69× |
| Max | 7.72× | 8.34× |
| Instances with ≥ 2× ratio | **90.3 %** | 73.0 % |
| Instances with ≥ 3× ratio | 46.5 % | 48.1 % |
| **Would be assigned to ≥ 2 different FPN levels across views** | **56.5 %** | 30.1 % |
| Per-camera median height range | 181 px (C6) → 344 px (C5), **1.9× between cameras** | 111 px (C4) → 186 px (C1), **1.7×** |

**[INF] Verdict: MSMVD's premise is real and quantitatively large.** On Wildtrack, more than half of all pedestrians would be handled at *different pyramid levels depending on which camera you look through*. A single-scale projection (MVDet, and CaMuViD's per-level-independent fusion) is structurally unable to represent that. This is the strongest data-backed motivation in your entire paper set — and it is *unimplemented*.

---

## 5. Previous approaches **[PAPER]** §2

| Category | Examples | What it does | Limitation |
|---|---|---|---|
| Monocular multi-scale | SSD, FPN, PAFPN, EfficientDet, Deformable DETR, RT-DETR, CSP, F2DNet, LSFM | Use features from several encoder depths; small objects on high-res levels | Single camera → no occlusion relief |
| Early MVPD (per-view detect → fuse) | Fleuret et al., POM, RCNN&clustering | Monocular detection per view, then aggregate results | Still relies on monocular detection, so occlusion persists |
| End-to-end BEV MVPD | **MVDet**, MVDeTr, SHOT, GMVD, 3DROM, MVAug, MVFP, OmniOcc | Project single-scale image features to BEV, predict one occupancy map | **Single-scale image features** — the gap |
| Improved projections | SHOT (stacked homographies), MVFP (non-parametric 3D feature pulling), 3DROM | Better geometry | Still single-scale |
| Data-augmentation line | MVAug, 3DROM, GMVD | Scene/view-level augmentation, generalisation | Orthogonal to scale |

---

## 6. Proposed method

```
 Iⁿ (N views, 720×1280)
   │
   ├─► ResNet-18 backbone  ──► {F₃ⁿ, F₄ⁿ, F₅ⁿ}   (strides 8, 16, 32)
   │                                │
   │                          image-FPN (PAFPN: top-down + bottom-up)
   │                                ▼
   │                        {F̃₃ⁿ, F̃₄ⁿ, F̃₅ⁿ}, C = 256
   │      small pedestrians better in F̃₃ⁿ · large pedestrians better in F̃₅ⁿ
   │
   ▼   ══════════ MSP: Multi-Scale Projection (scale-by-scale) ══════════
   for each level l ∈ {3,4,5}:
       for each height zᵢ ∈ {z₀..z₄}  (zᵢ = 30·i cm, 5 slices — 3DROM-style):
           warp F̃ₗⁿ to BEV at height zᵢ  using  γⁿ Kⁿ[Rⁿ|Tⁿ]
       concat the 5 height-warps along channels → 1×1 conv → Pₗⁿ ∈ ℝ^{C×X/2^{l-2}×Y/2^{l-2}}
                                                │
                              MAX-POOL over the N views (per level)
                                                ▼
                                   {B₃, B₄, B₅}   multi-scale BEV features
                                                │
   ══════════ BEV-FPN (PAFPN over BEV, top-down + bottom-up) ══════════
                                                ▼
                                   {B̃₃, B̃₄, B̃₅}
                                     │        │
                     sigmoid heads ──┤        ├── offset heads
                                     ▼        ▼
                              {M₃,M₄,M₅}   {O₃,O₄,O₅}
                                     │
              inference:  M = ⅓( M₃ + Up(M₄) + Up(M₅) ),  positions refined by O₃
```

### Module detail **[PAPER]** §3

| Module | In | Out | Operation | Purpose |
|---|---|---|---|---|
| Backbone | `3×720×1280` | `{F₃,F₄,F₅}` | ResNet-18, ImageNet-pretrained | multi-depth features |
| image-FPN | `{Fₗⁿ}` | `{F̃ₗⁿ}`, C=256 | **PAFPN** (top-down **and** bottom-up) | small pedestrians strong in F̃₃, large in F̃₅ |
| **MSP** | `{F̃ₗⁿ}` | `{Pₗⁿ}` | per level, per height slice: `γⁿ(uⁿ,vⁿ,1)ᵀ = Kⁿ[Rⁿ|Tⁿ](x,y,z,1)ᵀ` (Eq. 1); **`γⁿ` is set per level** so `Pₗⁿ` has resolution `X/2^{l−2} × Y/2^{l−2}` | **the core novelty** — the BEV feature *inherits the scale* of its source image level |
| View aggregation | `{Pₗⁿ}ⁿ₌₁ᴺ` | `Bₗ` | **max-pool along the view dimension**, per level | permutation-invariant, N-agnostic |
| **BEV-FPN** | `{B₃,B₄,B₅}` | `{B̃₃,B̃₄,B̃₅}` | PAFPN over BEV | mixes *scale* information *across* views |
| Heads | `B̃ₗ` | `Mₗ` (1ch, sigmoid), `Oₗ` (2ch) | conv | occupancy + sub-cell offset (MVDeTr-style) |

**[INF] Two ideas are being combined, and they are separable:**
- **MSP** = keep multiple scales *through* the projection (never collapse to one level before BEV).
- **BEV-FPN** = let scale information mix *after* view aggregation.
Their ablation (Table 3a) shows MSP alone is worth most of the gain (71.8 → 74.9 MODA), BEV-FPN adds the rest (→ 80.2).

---

## 7. Multi-view fusion strategy — the important part

**Mechanism: per-scale MAX-POOLING over views, then FPN mixing across scales.**

> `Bₗ ∈ ℝ^{C×X/2^{l−2}×Y/2^{l−2}}` is generated from `{Pₗⁿ}ⁿ₌₁ᴺ` by *"performing max pooling along the view direction, scale-by-scale"* **[PAPER]** §3.1.

**Why this helps — [INF]:**
1. **Permutation-invariant.** Max over the view axis; camera order is irrelevant. Contrast with MVDet's concat (P1) and CaMuViD's concat (P4), both order-dependent.
2. **N-agnostic architecture.** Max-pool over any N. **[INF] This is why MSMVD can train on GMVD at all** — GMVD scenes have *different numbers of cameras and different layouts*, which MVDet's `512N+2` first conv cannot express. This is a genuine architectural advantage, and the paper's GMVD result (80.2 vs 75.7 MODA for MVFP, **+4.5**) is its largest margin — much bigger than its Wildtrack margin (+0.5). **[INF] The gain is plausibly as much about N-agnosticism as about multi-scale.**
3. **Max is an implicit "best-view" selector.** For each BEV cell and channel, max picks the view with the strongest evidence — a crude but free form of view selection.

**Weaknesses — [INF]:**
1. **Max discards agreement.** Two views agreeing at 0.9 and one view at 0.9 are identical after max. Occlusion reasoning fundamentally *needs* to know *how many* views agree. ImGeoNet (P3) keeps mean **and variance**; MSMVD keeps neither.
2. **Max is dominated by outliers.** A false-positive activation in one bad view propagates unchallenged. There is no per-view confidence, visibility mask, or FoV normalisation described.
3. **Still uniform across views** — no learned view weighting. Every view competes on raw activation magnitude.
4. **Height slices are hand-set** (`zᵢ = 30·i` cm, 5 slices) — a fixed prior about pedestrian body extent that will not transfer to arbitrary object categories.
5. **Not evaluated for missing views.** No camera-dropout study.

---

## 8. Loss function **[PAPER]** Eqs. 2–3

```
L_all = Σ_{l∈{3,4,5}} [ L_det(Mₗ, M̂ₗ) + L_off(Oₗ, Ôₗ) ]        (2)
```
- `L_det` = **Focal loss** (Lin et al.) on the occupancy maps.
- `L_off` = **L1 variant** on the offset maps.
- `M̂ₗ` = GT occupancy down-sampled to `Mₗ`'s resolution then Gaussian-smoothed; `Ôₗ` = the discretisation error of `M̂ₗ`.
- No `λ` weights: **all terms weight 1**.
- `O₄`, `O₅` are computed **only as auxiliary losses** and are *not* used at inference.

Inference merge (Eq. 3): `M = ⅓{M₃ + Up(M₄) + Up(M₅)}`, then threshold **0.4**, then refine with `O₃`.

**[INF]** Note the design asymmetry: multi-scale in the *loss* (deep supervision at 3 resolutions) and multi-scale at *inference* (averaging 3 maps). Table 3b isolates this: using `M₃&O₃` alone gives 79.1 MODA vs 80.2 for the merge — so **the merge is worth ~1.1 MODA**, and the *representation* (not the merging trick) carries the rest.

---

## 9. Datasets **[PAPER]** §4.1

| | Wildtrack | MultiviewX | **GMVD** (main dataset) |
|---|---|---|---|
| Cameras | 7 | 6 | **varies per scene** |
| Area | 12 × 36 m | 16 × 25 m | as MultiviewX |
| Grid | 480 × 1440 (2.5 cm) | 640 × 1000 (2.5 cm) | as MultiviewX |
| Frames | 400 | 400 | 4983 train (6 scenes) + 1012 test (1 scene) |
| People/frame | 20 | 40 | 20–40 |
| Split | 360 train / 40 test | 360 / 40 | 6 scenes train / **1 unseen scene test** |
| Validation | 40 random frames from train | same | **MultiviewX used as val** |

**[INF] GMVD is the important one.** It is the only dataset here that tests **generalisation to an unseen camera layout** — the thing all fixed-N methods fail at. That MSMVD leads by +4.5 MODA there but only +0.5 on Wildtrack/MultiviewX is the single most informative number in the paper for your research: *scale-aware, N-agnostic fusion matters most exactly when the camera configuration changes.*

**Augmentation:** random resizing and cropping, following MVDeTr.

---

## 10. Experimental setup **[PAPER]** §4.2

| Item | Value |
|---|---|
| Backbone | ResNet-18 (ImageNet-pretrained); also R34/R50/R101 in Table 4 |
| Image-FPN / BEV-FPN | PAFPN, C = 256 |
| Input | 720 × 1280 |
| Optimiser | Adam |
| LR | 1.0e−3 → 1.0e−6, **cosine** schedule |
| Batch | **1**, with **gradient accumulation over 16 batches** (effective 16) |
| Epochs | **10** on GMVD, **50** on Wildtrack/MultiviewX |
| Detection threshold | 0.4 |
| Prediction heads | 4 conv layers each |
| Metrics | MODA (primary), MODP, precision, recall; TP if within **0.5 m** |
| GPU / CPU / CUDA / PyTorch version | ❌ **Not reported.** No hardware, no library versions, no runtime, no memory. |

**[INF]** The omission of hardware and runtime is a real reproducibility deficit, compounded by the absence of code. Params are the only cost figure given (Table 4): 22.9 M (R18) → 54.9 M (R101), vs baseline 11.7 M → 43.2 M — i.e. **MSP + BEV-FPN cost ~11.2 M extra parameters**, constant across backbones.

---

## 11. Attempt to execute — not applicable

| Problem | Cause | Solution attempted | Result |
|---|---|---|---|
| No code | Repository not released and not present in `P2/` | Listed folder recursively; searched whole tree for MSMVD/BEV-FPN/MSP sources | ❌ Only the PDF exists |
| Cannot verify any claim against source | — | — | ❌ All claims remain **[PAPER]**-only |
| Cannot reproduce any number | — | — | ❌ |

**Nothing was run.** Additionally, this machine has **no PyTorch and no CUDA GPU** (verified: `import torch` → `ModuleNotFoundError`; `nvidia-smi` → not found), so even a released repo could not have been run here.

---

## 12. Reported results (paper claims, unverifiable)

**Table 1 — GMVD** (†= MSMVD authors' re-implementation)

| Method | MODA | MODP | Prec | Rec |
|---|---|---|---|---|
| Vora+ | 68.2 | 76.3 | 91.5 | 75.5 |
| SHOT† | 71.5 | 77.9 | 93.7 | 76.7 |
| Suzuki+† | 72.3 | 77.1 | 91.3 | 77.1 |
| 3DROM† | 73.7 | 77.3 | 92.2 | 80.5 |
| MVAug† | 73.8 | 77.9 | 91.9 | 80.8 |
| OmniOcc† | 75.1 | 76.9 | 92.3 | 82.0 |
| MVFP† | 75.7 | 78.2 | 94.3 | 80.5 |
| **MSMVD** | **80.2** | **81.3** | **95.7** | **83.9** |

**Table 2 — Wildtrack / MultiviewX**

| Method | WT MODA | WT MODP | MVX MODA | MVX MODP |
|---|---|---|---|---|
| MVDet | 88.2 | 75.7 | 83.9 | 79.6 |
| MVDeTr | 91.5 | 82.1 | 93.7 | **91.3** |
| MVAug | 93.2 | 79.8 | 95.3 | 89.7 |
| 3DROM | 93.5 | 75.9 | 95.0 | 84.9 |
| MVFP | 94.1 | 78.8 | 95.7 | 85.1 |
| **MSMVD** | **94.6** | **83.3** | **97.2** | **91.3** |

**Ablations**

| Table 3a | MODA | | Table 3b (inference merge) | MODA |
|---|---|---|---|---|
| Baseline (single-scale) | 71.8 | | M₃&O₃ only | 79.1 |
| + MSP | 74.9 | | M₄&O₄ only | 78.1 |
| + MSP + BEV-FPN | **80.2** | | M₅&O₅ only | 77.6 |
| | | | merged (Ours) | **80.2** |

| Table 4 — backbone | Baseline MODA / Params | MSMVD MODA / Params |
|---|---|---|
| ResNet18 | 71.8 / 11.7 M | **80.2 / 22.9 M** |
| ResNet34 | 72.1 / 21.8 M | 81.0 / 33.0 M |
| ResNet50 | 74.3 / 24.2 M | 82.0 / 35.9 M |
| ResNet101 | 75.8 / 43.2 M | **82.2 / 54.9 M** |

| Metric | Paper reported | My reproduction | Difference | Explanation |
|---|---|---|---|---|
| GMVD MODA | 80.2 | **not attempted** | — | **No code released.** Not a resource limitation |
| Wildtrack MODA | 94.6 | not attempted | — | " |
| MultiviewX MODA | 97.2 | not attempted | — | " |

---

## 13. Critical reading — **[INF]**

**Strong points**
- The **parameter control (Table 4) is the right experiment** and it is convincing: MSMVD-R18 (22.9 M, 80.2) beats baseline-R101 (43.2 M, 75.8). The gain is *not* just capacity. Few papers do this; credit where due.
- Ablations are clean and additive (71.8 → 74.9 → 80.2).
- The GMVD-vs-Wildtrack margin gap (+4.5 vs +0.5) is honestly reported and is the most scientifically interesting result.

**Weak points**
1. **No code, no hardware, no runtime, no FPS.** For a method that adds 3 BEV pyramids and 5 height slices per level (**15 warps per view per frame** vs MVDet's 1), the absence of any latency/memory number is a significant omission. **[INF]** MSP is plausibly the most expensive projection stage in your entire paper set, and it is the only one with no cost reported.
2. **Wildtrack/MultiviewX gains are within noise territory.** +0.5 MODA over MVFP on 40 test frames (~800 pedestrians). No error bars, no seed variance, single run. The GMVD gain (+4.5, 1012 frames) is the credible one.
3. **Six of the eight GMVD baselines are the authors' own re-implementations (†).** Comparisons against re-implementations are weaker evidence than against author-released numbers.
4. **`γⁿ` per-level scaling is under-specified.** §3.2 says `γⁿ` "accounts for a down-sampling of image features and determines the spatial resolution of the BEV feature", set so `Pₗⁿ` matches the image level's resolution ratio. Without code, the exact construction of the sampling grid is not reconstructable with confidence — **[INF] this is the single hardest part to re-implement correctly.**
5. **No missing-view / camera-dropout study**, despite max-pooling being naturally robust to it. A missed opportunity that you could fill.

---

## 14. Failure analysis

No predictions, no logs, no code → this section is necessarily thinner.

| Failure type | Evidence | Likely cause | Potential fix |
|---|---|---|---|
| Max-pool loses cross-view agreement | **[INF]** from Eq./§3.1 | `max` over views is a 1-of-N selector; multiplicity is discarded | Add variance/count channels (ImGeoNet does exactly this) |
| Single bad view can dominate | **[INF]** | No per-view confidence or FoV mask described | Gate by predicted per-view visibility before pooling |
| Fixed height slices `z = 0,30,60,90,120` cm | **[PAPER]** §3.2 | Hand-tuned to human bodies | Learn slice positions, or use a voxel grid, if you generalise past pedestrians |
| Small gains on fixed-rig datasets | **[PAPER]** Tables 1 vs 2 | Wildtrack/MVX are near-saturated (94–97 MODA); little headroom | Evaluate on GMVD-style varying rigs |
| **[OBS]** Its own premise is *stronger* on Wildtrack (56.5 % multi-level) than MultiviewX (30.1 %) | my measurement, §4 | Wildtrack's cameras are at more varied distances | Yet MSMVD gains *more* on MultiviewX (+1.5) than Wildtrack (+0.5) — **[INF] an inconsistency the paper does not address**, suggesting scale is not the only driver |

---

## 15. Limitations

**Methodological** — no view weighting; max-pool discards agreement/count; hand-set height slices; no explicit occlusion or visibility model; no view selection.
**Dataset** — Wildtrack/MVX are 400-frame, near-saturated benchmarks; GMVD is synthetic.
**Computational** — 15 warps/view/frame; +11.2 M params; **cost never measured**.
**Generalisation** — good sign on GMVD, but only one held-out scene; no cross-dataset (train MVX → test WT) result, which P4 and others do report.
**Implementation** — **no code, no hardware spec, no runtime, no seed/variance.** By the standard your own project sets ("prioritise reproducibility over speed"), P2 is currently the least reproducible work in your set.

---

## 16. Research gaps

**Gap 1 — Scale-aware fusion exists, but view-aware fusion does not.**
MSMVD makes fusion *scale-adaptive* and leaves it *view-uniform* (a plain max). The two axes are independent and only one has been addressed.
> **Direction:** a fusion operator that is adaptive on **both** axes — for each BEV cell, choose *which scale* and *which views* to trust, e.g. attention over the (view × scale) set rather than max over views then FPN over scales.

**Gap 2 — Max-pooling throws away the evidence-count signal that occlusion reasoning needs.**
**[OBS]** On Wildtrack, pedestrians are seen by a mean of 5.26 of 7 cameras, ranging from 1 to 7. Whether a BEV cell is supported by 1 view or 7 is highly informative — and `max` erases it.
> **Direction:** fuse with a **set-statistics vector** (max, mean, variance, valid-count) per cell per scale. Cheap, N-agnostic, permutation-invariant, and strictly more informative than max. **[INF] This is a small, well-scoped, defensible contribution.**

**Gap 3 — The cost of multi-scale projection is unquantified.**
> **Direction:** measure it. A latency/accuracy Pareto curve over {#scales, #height slices} would be a genuine contribution, because the paper provides none.

**Gap 4 — Multi-scale BEV has never been combined with N-varying camera rigs *and* missing views.**
GMVD varies N across scenes but always provides all of them.
> **Direction:** train once on GMVD, evaluate under random camera dropout. Max-pool should degrade gracefully where MVDet cannot even run — a clean, publishable comparison.

---

## 17. What to borrow

| Useful idea | Why it works | Limitation | How to adapt for your research |
|---|---|---|---|
| **Project every scale, don't collapse first** (MSP) | The BEV feature *inherits* the scale properties of its source image level, so small objects stay on high-res BEV and large on low-res. Worth **+3.1 MODA** alone | 15 warps/view; cost unmeasured | **Highest-value idea in your set.** **[OBS]** justified: 56.5 % of Wildtrack pedestrians span multiple FPN levels across views |
| **Max-pool over views, per scale** | Permutation-invariant and N-agnostic → the architecture survives a changed camera rig, which MVDet's concat cannot | Discards agreement/count | Keep the invariance, **replace `max` with richer set statistics or attention** |
| **FPN *in BEV space*** (BEV-FPN) | Lets scale information mix *after* views are merged, so a large-scale cue from view 2 can inform a small-scale cue from view 5. Worth **+5.3 MODA** | Extra params | Directly transferable to any BEV/voxel representation |
| **Deep supervision at multiple BEV resolutions + inference merging** | Regularises each scale; merging adds ~1.1 MODA | Modest | Cheap to add to any BEV head |
| **Parameter-matched backbone sweep (Table 4)** | The correct way to prove a gain isn't just capacity | — | **Copy this experimental protocol.** Reviewers will ask; MSMVD's answer is the template |
| **Using GMVD as the main benchmark** | Only dataset that varies camera count/layout, so it actually tests view generalisation | Synthetic | **Adopt it.** It is where your view-adaptive ideas will show their largest margin |

---

## 18. Verdict

**Conceptually the most useful paper in your set; practically the least usable.**

- **Do not** plan to use P2 as a baseline — there is nothing to run, and re-implementing MSP correctly from the text alone is the riskiest engineering task available to you (§13.4).
- **Do** take three things: (i) **multi-scale projection** — empirically justified by my own measurement, (ii) **permutation-invariant, N-agnostic view pooling**, (iii) the **GMVD benchmark + parameter-matched ablation protocol**.
- **Do** treat "max-pool discards cross-view agreement" (Gap 2) as a concrete, low-risk improvement you can propose and test.

**Cross-references:** P1 (MVDet) is the single-scale ancestor it improves · P3 (ImGeoNet) already uses the mean+variance statistics P2 lacks · P5 (CVT) already has the learned view weighting P2 lacks. **[INF] The combination none of the six papers has is: multi-scale (P2) × set-statistics (P3) × learned view weighting (P5). That intersection is where your contribution can sit.**
