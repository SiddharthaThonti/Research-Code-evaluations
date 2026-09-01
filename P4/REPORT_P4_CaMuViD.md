# P4 — CaMuViD: Calibration-Free Multi-View Detection

**Analysis date:** 2026-08-29
**Evidence base:** paper PDF (CVPR 2025, pp. 1220–1229) + full source (`CaMuViD/`) + the **COCO annotation files shipped in `data/`**, which I used for original measurements
**Execution status:** ❌ not executed (§11) — but this folder shipped real ground-truth data, and I ran **three original analyses** on it.

Tags: **[PAPER]** claimed · **[CODE]** verified in source · **[OBS]** measured by me this session · **[INF]** my inference

---

## 1. Inventory

| Item | Value |
|---|---|
| Title | CaMuViD: Calibration-Free Multi-View Detection |
| Authors | Amir Etefaghi Daryani, M. Usman Maqbool Bhutta, Byron Hernandez, Henry Medeiros (University of Florida, AI Laboratory) |
| Venue / Year | **CVPR 2025**, pp. 1220–1229 |
| Code | `CaMuViD/` — InternImage/MMDetection 2.28.1 fork; core logic in `camuvid.py`, `Custom_TwoStageDetector.py` (978 lines), `config.py`, `evaluation.py` |
| README | Yes (9.3 KB), unusually detailed with full result tables |
| Datasets | Wildtrack, MultiviewX. **Images absent** (`data/{DS}/Image_subsets/` does not exist) |
| **Shipped data** | ✅ **8 COCO-format annotation JSONs (25 MB)** — train/test × 2 datasets × {with, without} person IDs. **Real ground truth.** |
| Pretrained weights | HuggingFace links in README (`multiviewx.pth`, `wildtrack.pth`) — **not on disk** |
| Config | `config.py` (a Python class, `Config_d`) + `configs/{DS}/*.py` mmdet configs |
| Train / eval | both via `python camuvid.py`, switched by `Config_d.mode = "train"|"test"` |
| Requirements | README install block: torch 1.11+cu113, mmcv-full 1.5.0, mmdet 2.28.1, mmsegmentation 0.27.0, timm 0.6.11, numpy<2, **custom DCNv3 CUDA op** |
| Result files | None (no logs, no prediction dumps) |
| Reproducibility | **Blocked**: no images, no weights, no GPU; DCNv3 needs `nvcc` |

---

## 2. Relevance to *Multi-View Object Recognition*

**A. Problem solved.** Detect pedestrians across *N* overlapping camera views **without using camera calibration**, producing 2D boxes **in each camera's own image space** (not a BEV occupancy map).

**B. What "multi-view" means here.** **Multiple fixed cameras** of one scene. *N* is fixed and hard-wired (7 Wildtrack / 6 MultiviewX). Not multi-viewpoint object capture; not active view selection.

**C. Input.** `I = {I_i}ᴺ₌₁`, `I_i ∈ ℝ^{3×H×W}`, resized to **640 × 1338** (`config.py:61-62`). **No intrinsics/extrinsics are used by the model** — calibration XMLs are listed in `config.py` but only for *evaluation* (projecting unmatched boxes to world coords).

**D. Output.** `B = {B_i}ᴺ₌₁` — a set of 2D pedestrian bounding boxes **per camera view**. **No BEV map, no 3D position, no category** (single class).

**E. Relevance score.**

| Reading | Score | Why |
|---|---|---|
| *Multi-camera scene recognition* | **3 / 5** | Same task family and datasets; the **calibration-free premise** is genuinely valuable and rare |
| *Multi-viewpoint object recognition* | **2 / 5** | Output is per-view 2D boxes; nothing about object identity across viewpoints or 3D understanding |

**Net: 2.5 / 5.** ⚠ **Lower than its CVPR-2025 headline suggests**, for two concrete reasons I verified in code (§7, §12): the "projection" performs **no spatial alignment**, and the **evaluation protocol differs from every method it is tabulated against**. Use it as an *idea source for calibration-free fusion*, not as a baseline.

---

## 3. Research problem

- **Input:** N synchronised RGB views. **No calibration.**
- **Output:** per-view 2D boxes.
- **Assumptions:** cameras synchronised; **fixed, known N** (baked into the architecture); each camera's viewpoint is *static across train and test* (so the network can memorise its scene layout).
- **Objective:** maximise MODA under the authors' image-space evaluation.

---

## 4. Motivation and the gap claimed **[PAPER]**

1. **Calibration is a burden and a failure point.** Inverse projection to a BEV ground plane "distorts features" for distant pedestrians, "complicating the extraction of reliable features from targets farther from the camera".
2. **BEV methods overfit the rig.** Methods that memorise a single ground-plane structure (MVDet, MVAug) collapse when the camera configuration changes — cross-dataset MODA drops to 17.0 / 26.3 **[PAPER]** Table 2.
3. **BEV loses the image-space evidence.** Detecting in image space keeps appearance information that BEV projection destroys.

**Their proposal:** learn projection `A^p` and back-projection `A^b` matrices *from the features themselves*, constrained to be inverses, and detect in each view's own image space.

---

## 5. Previous approaches **[PAPER]** §2

| Category | Examples | Limitation named |
|---|---|---|
| Monocular detection | Faster R-CNN, FPN, SSD, YOLO, DETR, CenterNet, FoveaBox | Occlusion, scale variation, no cross-view reasoning |
| CRF / mean-field multi-view | POM, Baqué et al., Roig et al. | Hand-designed potentials |
| Calibrated BEV projection | **MVDet**, SHOT (stacked homographies), MVDeTr (deformable attention + view augmentation), MVFP (non-parametric 3D feature pulling) | Depend on calibration; inverse projection distorts and produces shadow artefacts |
| Augmentation for generalisation | MVAug, 3DROM, GMVD | "poor generalization, tending to overfit to specific scenes and camera configurations" |

---

## 6. Proposed method

```
 I₁..I_N ──► InternImage-T backbone D (COCO-pretrained, DCNv3) + FPN
                       │
              F = {F_i}, each a 4-level FPN pyramid, C_f = 256
                       │
     ┌─────────────────┴──────────────────┐
     │  A^p_i = N_p(F_i)  ∈ ℝ^{256×256}   │  ← FCN: GAP → 256→128→256→256·256, reshape
     │  P_i   = A^p_i × F_i               │  (Eq. 2)
     └─────────────────┬──────────────────┘
                       │  per FPN level, per view
              P_c = [P_1, P_2, ..., P_N]        (Eq. 3) CHANNEL CONCAT  → 256·N
                       │
              P_f = N_c(P_c)     1×1 Conv2d(256N → 256)   (Eq. 4)
                       │
     ┌─────────────────┴──────────────────┐
     │  A^b_i = N_b(F_i)                  │  (Eq. 5)
     │  F̂_i  = A^b_i × P_f                │  (Eq. 6)
     │  constraint  A^b_i = (A^p_i)⁻¹     │  enforced by L_vp (Eq. 7)
     └─────────────────┬──────────────────┘
                       │
              FRM_i (Conv7×7 → ReLU → Conv3×3 → ReLU)   per view
                       │
              DetHead_i  (Cascade R-CNN, per view)      → B_i
```

### Module detail **[CODE]** `Custom_TwoStageDetector.py`

| Module | Where | What it actually does |
|---|---|---|
| `ProjectionMatrixNetwork` | L382–416 | `GAP(F_i)` → Linear(256→128) → ReLU → Linear(128→256) → ReLU → Linear(256→**65 536**) → reshape to **256×256** |
| `projection_net_1` / `_2` | L523–525 (`for i in range(2)`) | **Two** networks total, shared across all views. `_1` produces `A^p`, `_2` produces `A^b` |
| apply projection | L681–683 | `flatten = F_i.view(B,C,H*W)` ; `P_i = torch.matmul(A^p, flatten).view(B,C,H,W)` |
| fusion | L716 | `torch.cat([f[i] for f in projected_features], dim=1)` |
| channel reduce | L530, L719 | `self.conv = nn.Conv2d(256*num_head, 256, kernel_size=1)` |
| back-projection | L741 | `F̂_i = torch.matmul(A^b_i, P_f_flat).view(B,C,H,W)` |
| `Refiningmodule_i` | L519 | **one per view** |
| `rpn_head_{i+1}`, `roi_head_{i+1}` | L505, L513 | **one Cascade R-CNN head per view** |

---

## 7. ⚠ Critical code finding: the "projection" performs **no spatial transformation**

This is the most important technical observation in this report.

**[CODE]** `A^p_i ∈ ℝ^{256×256}` is produced from `GAP(F_i)` — a **global** descriptor with all spatial information pooled away (L405–406). It is then applied as

```python
flatten_feature = features[i][j].view(batch, C, H*W)        # (B, 256, HW)
projected_feature = torch.matmul(attention_map1, flatten_feature).view(batch, C, H, W)
```

`matmul((B,256,256), (B,256,HW)) → (B,256,HW)`.

**[INF] Therefore `A^p_i` is a channel-mixing matrix applied *identically at every spatial location*.** It cannot move, warp, rotate or resample anything. The "common representation space" is a **channel space**, not a geometric one.

Consequences, all verifiable from the same lines:

1. **Fusion aligns views by raster index, not by geometry.** `torch.cat` at L716 concatenates `P_1[:, :, h, w]` with `P_2[:, :, h, w]` — pixel *(h,w)* of camera 1 with pixel *(h,w)* of camera 2. Those two pixels view **completely unrelated parts of the scene**. The subsequent `Conv2d(256N, 256, kernel_size=1)` is **1×1**, so it has *no* spatial receptive field to repair the misalignment either.
2. **The model must memorise each camera's static scene layout** in its per-view `refiningmodule_i` / `rpn_head_i` / `roi_head_i`. This works *only because Wildtrack and MultiviewX cameras never move between train and test.*
3. This reframes the paper's own Fig. 5 comparison. **[PAPER]** contrasts "geometry-based" (warp to world plane and back) against "geometry-free" and argues the geometric route "loses spatial details... causing misalignment across views". **[INF] What the released code shows is not a *different* alignment — it is *no* alignment.** The gain over the geometric variant (−9.1 MODA when `A^p/A^b` are replaced by camera-parameter projections, **[PAPER]** §4.6) is real, but its most plausible explanation is that per-view heads on *unwarped* features beat shared heads on *warped, resampled, artefact-laden* features — not that a learned "projection" found a better common space.

**[INF] Fair reading:** CaMuViD is best described as **N parallel detectors with a shared, channel-mixed feature bottleneck**, not as a calibration-free multi-view *projection* method. That is still a legitimate and effective design — but the mechanism differs from the paper's framing.

---

## 8. Multi-view fusion strategy

**Mechanism: per-view channel-mixing → CHANNEL CONCAT across views → 1×1 conv → per-view channel-mixing back.**

**Why it helps (paper's argument, and it is a good one)** **[PAPER]** §3.3:
> *"The main benefit of concatenation over summation is that it does not dilute information from distinct views. That is, a high-activation area summed to a low-activation area would force the two activations to the middle, which is undesirable."*

Their ablation supports it (**[PAPER]** Table 4): Sum + no FRM = 93.8 MODA → Concat + no FRM = 94.6 → Concat + FRM = **95.0**.

**Weaknesses — [INF]:**

1. **No spatial alignment at all** (§7). This is the dominant limitation.
2. **Architecture hard-wired to N.** `nn.Conv2d(256*num_head, 256, 1)` plus N copies of RPN/ROI/refining heads. Changing N requires rebuilding and retraining. **[CODE]** `config.py:12-13` — you must manually set `num_head` and `num_views` to 7 or 6.
3. **Not permutation-invariant.** Concat order fixes camera identity in channel slots, and the per-view heads make it *doubly* order-dependent.
4. **Parameter cost is severe.** Each `ProjectionMatrixNetwork` ends in `Linear(256 → 65 536)` ≈ **16.8 M params**; two of them ≈ 33.6 M. Plus **N full Cascade R-CNN heads**. **[INF] At N=7 the detection heads alone dominate the model.**
5. **Dead code creates confusion.** L499 loops `for i in range(self.num_head)` creating per-view modules; L523 loops `for i in range(2)` creating exactly two projection nets; but L679/L693 call `projection_net_1`/`_2` unconditionally. The per-view projection nets implied by Eq. 1's subscript `i` are **shared networks**, not per-view ones. (The *matrices* still differ per view, since inputs differ — so Eq. 1 is satisfied — but the paper's notation invites the wrong reading.)

---

## 9. Loss function **[PAPER]** Eq. 7

```
L_vp = Σ_{i=1}^{N} | A^b_i × P_i − F_i |          (view projection / cycle-consistency loss)
L    = L_det(bbox + cls, Cascade R-CNN)  +  1e-4 · L_vp
```

**[PAPER]** §3.5: *"we use the view projection loss (Eq. 7) in addition to standard object detection loss terms (bounding boxes and classification losses) with a weight of 1e−4."*

**[INF] The weight is 1e−4 — four orders of magnitude below the detection loss.** The inverse constraint `A^b_i = (A^p_i)⁻¹` is therefore a very weak regulariser, not a hard constraint. Combined with §7, this means the model is under almost no pressure to make the projections meaningfully invertible. **No ablation on this weight is reported** — an obvious missing experiment.

---

## 10. Datasets

| | Wildtrack | MultiviewX |
|---|---|---|
| Cameras | 7 | 6 |
| Area / grid | 12×36 m, 480×1440 @ 2.5 cm | 16×25 m, 640×1000 @ 2.5 cm |
| Images | 1080×1920 → **640×1338** | same |
| People/frame | ~20 | ~40 |
| Avg cameras per grid cell | 3.74 | 4.41 |
| Split | first 90 % train / last 10 % test | same |
| Epochs / LR | 20 / 1e−4, milestones [5,10,50,75], γ=0.5 | same |
| Batch | 1 | 1 |

### **[OBS] My original measurements on the shipped GT** (`research_analysis/09_scripts/view_coverage.py`)

**Wildtrack test — 40 frames, 1001 (frame, person) instances, 5276 boxes**

| Cameras seeing one person | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|
| share | 2.0 % | 2.8 % | 7.9 % | 5.9 % | 26.5 % | **45.6 %** | 9.4 % |

Mean **5.26 of 7** cameras per person.

| Camera | C1 | C2 | C3 | C4 | C5 | C6 | C7 |
|---|---|---|---|---|---|---|---|
| % of instances seen | 91.9 | 83.4 | 92.8 | **27.7** | 71.9 | **96.3** | 62.2 |
| instances only this camera sees | 7 | 0 | 4 | 0 | 0 | 5 | 4 |
| mean box area (px) | 21 536 | 12 528 | 30 878 | 32 911 | 39 724 | 10 233 | **40 595** |

**Greedy coverage order: C6 → C3 → C1 → C7 → C2 → C4 → C5**, reaching **99.50 % with 3 cameras** and **100 % with 4**. Cameras **C2, C4 and C5 each add exactly 0.00 pp** once {C6,C3,C1,C7} are present. Pairwise Jaccard overlap C1↔C6 = 92.8 %, C3↔C6 = 92.0 %.

**MultiviewX test — 1494 instances:** mean 4.97 of 6 cameras; **zero** instances seen by only one camera; **C3+C4 alone give 100 % coverage**.

**[INF] Both benchmarks are heavily over-covered.** Beyond ~3–4 cameras, additional views add essentially **zero new pedestrians** — so any accuracy gain from views 4–7 must come from *disambiguation under occlusion*, not from coverage. That distinction is what §12 makes measurable.

---

## 11. Attempt to execute — failed, documented

```
OS      Windows 11        Python 3.14.3 (conda base only)
PyTorch NOT INSTALLED     GPU: NONE (nvidia-smi not found)
```

| Problem | Cause | Solution attempted | Result |
|---|---|---|---|
| No PyTorch / no CUDA GPU | Machine has neither | `import torch`; `nvidia-smi` | ❌ |
| **DCNv3 custom CUDA op** | InternImage backbone needs `cd ops_dcnv3 && sh make.sh` (or a prebuilt `.whl`) | Read README + confirmed `ops_dcnv3/` present with `DCNv3.egg-info` | ❌ Needs `nvcc` + matching torch CUDA |
| mmcv-full 1.5.0 / mmdet 2.28.1 | Pinned, old; Windows wheels for that combo are not published for modern Python | Read README install block | ❌ |
| **No images** | `data/{DS}/Image_subsets/` absent — only the annotation JSONs | `Get-ChildItem data -Recurse` | ❌ Wildtrack/MultiviewX must be downloaded separately |
| No weights | HuggingFace links only | Searched tree for `*.pth` | ❌ Zero checkpoints |
| Needs backbone checkpoint too | `config.py:38` `pretrained_backbone = 'checkpoint/fine_tune_{DS}_chunk_t_3x/best_bbox_mAP_epoch_10.pth'` | Read `config.py` | ❌ Also absent — a **two-stage** pipeline (finetune backbone, then train fusion) |
| Hard-coded 2-GPU-ish assumptions | `config.py:14` `gpu_num = 2` | Read `config.py` | ⚠ |

**Nothing was run.** The three analyses I *did* run (§10, §12, and the scale study) use **only the shipped annotation JSONs** and pure-Python — no model, no GPU.

---

## 12. ⭐ Original experiment: CaMuViD's camera-elimination ablation vs. the data's oracle ceiling

### 12.1 First, what the ablation actually does **[CODE]**

The README/paper Table 3 "camera elimination" is **not** training with fewer cameras. It is:

```python
# camuvid.py L243-248  (and identically custom_datasets_fn.py L132-137)
if blank_views == None or positive_pair_camera not in blank_views:
    positive_pair_image = dataset.load_image(pair_image_path_p)
elif positive_pair_camera in blank_views:
    positive_pair_image = Image.new('RGB', (original_shape[1], original_shape[0]), (0,0,0))
```

The removed camera is fed a **black image**; *N* and the architecture are unchanged; the model is **not retrained**. Further, **[CODE]** L328–330:

```python
for j, view in enumerate(camera_views_list):
    # if view in blank_views:
    #     continue                    <-- SKIP IS COMMENTED OUT
```

so the **ground truth of blanked cameras is still scored**. A blanked camera can therefore only produce false negatives.

**[INF] Correct interpretation:** Table 3 measures **robustness to missing/corrupted views at test time**, not **scaling with camera count**. Those are different questions, and the paper's phrasing ("By integrating five cameras, we recover to 95.6 % MODA") reads as the latter.

### 12.2 The comparison

Because the TP rule is per-identity — **[PAPER]** §4.2: *"If an ID is assigned to at least one detection, it is counted as a TP; otherwise, it is considered an FN"* — the maximum achievable recall for a kept-camera subset S is exactly

`oracle(S) = |{(frame,pid) visible in ≥1 camera of S}| / |all (frame,pid)|`

which I compute directly from the shipped GT (`research_analysis/09_scripts/camuvid_ablation_check.py`):

| k | cameras kept | **oracle recall** | CaMuViD recall | achieved / oracle | CaMuViD MODA |
|---|---|---|---|---|---|
| 1 | C1 | 91.91 % | 60.2 % | 65.5 % | 60.1 |
| 2 | C1+C2 | 93.31 % | 77.9 % | 83.5 % | 77.8 |
| 3 | +C3 | 98.50 % | 92.1 % | 93.5 % | 90.6 |
| 4 | +C4 | 98.50 % | 95.5 % | 97.0 % | 93.8 |
| 5 | +C5 | 98.50 % | 96.9 % | 98.4 % | 93.8 |
| 6 | +C6 | 99.60 % | 99.3 % | 99.7 % | **95.6** |
| 7 | +C7 | 100.00 % | 98.6 % | 98.6 % | 95.0 |

Marginal analysis:

| added | Δ oracle coverage | Δ CaMuViD recall | Δ MODA |
|---|---|---|---|
| C2 | +1.40 pp | +17.7 pp | +17.7 |
| C3 | +5.19 pp | +14.2 pp | +12.8 |
| **C4** | **+0.00 pp** | **+3.4 pp** | +3.2 |
| **C5** | **+0.00 pp** | **+1.4 pp** | +0.0 |
| C6 | +1.10 pp | +2.4 pp | +1.8 |
| **C7** | **+0.40 pp** | **−0.7 pp** | **−0.6** |

### 12.3 Three findings — [INF], each grounded in the table above

**Finding 1 — Cameras 4 and 5 help *purely* through occlusion disambiguation.**
They add **zero** new pedestrians (oracle stays at 98.50 %), yet recall rises **+3.4** and **+1.4 pp**. This is the cleanest available evidence, in your whole paper set, that redundant views carry real value — but *only* as corroborating evidence against occlusion, not as coverage. Every method here (MVDet's concat, ImGeoNet's mean, MSMVD's max, GeoBEV's sum) aggregates these two kinds of contribution identically, with no mechanism to distinguish them.

**Finding 2 — The 7th camera is net-harmful.**
C7 adds +0.40 pp of oracle coverage, yet recall **drops 0.7 pp**, precision drops (96.6 → 96.3) and MODA drops **0.6**. The 6-camera configuration is the paper's own best result (95.6 MODA), and the headline number (95.0) is *worse*. The README states this plainly; the paper's text does not comment on it.
**[OBS] Corroboration:** C7 has the **largest mean box area (40 595 px)** and only 62.2 % coverage — it is a close-up, narrow-FoV camera, prone to generating detections that duplicate other views. Under a fixed-N concat architecture, it cannot be down-weighted.
> **This is the single strongest published data point for the research gap "more views is not monotonically better".**

**Finding 3 — Above 3 cameras, CaMuViD is essentially saturating the data.**
Achieved/oracle rises 65.5 % → 93.5 % → 99.7 %. From k=4 onward, remaining error is **not** a fusion problem — the ceiling is nearly reached. Improvements must come from the k≤3 regime (single/dual-view occlusion) or from *precision* (MODA still trails recall by ~4 points).

**[OBS] Train/test asymmetry worth noting:** on the **train** split, C1+C2+C3 already gives **100.00 %** oracle coverage (8566 instances); on **test**, 3 cameras give only 98.50 % and you need all 7 for 100 %. The test frames are genuinely harder to cover.

---

## 13. ⚠ Evaluation-protocol comparability

**[CODE]** `evaluation.py:43-116`: TPs are assigned by **2D image-space bounding-box IoU ≥ 0.45** with Hungarian matching (`box_iou` + `linear_sum_assignment(-iou)`), per camera; unmatched predictions are then projected to world coordinates (L145–170) and clustered before being counted as FPs.

Every method in Table 1 that CaMuViD is compared against (MVDet, SHOT, MVDeTr, MVAug, 3DROM, MVFP, TrackTacular) computes MODA on the **ground plane with a 0.5 m Euclidean distance threshold**.

**To be fair, the paper discloses this** **[PAPER]** §4.2: *"our approach only produces bounding boxes... there is a subtle but important distinction in the determination of True Positives... our MODP is computed based on IoU. In contrast, BEV-based approaches compute Euclidean distances in the world plane, which makes our MODP more strict."*

**[INF] But the disclosure is framed around MODP only, while MODA, precision and recall are equally affected**, and Table 1 juxtaposes all five metrics with no marker distinguishing the protocols. A per-identity "detected in ≥1 of 7 views" TP rule is structurally more permissive on recall than a single ground-plane point match. **Conclusion: CaMuViD's 95.0 vs MVFP's 94.1 is not a like-for-like comparison.** If you use CaMuViD as a baseline, you must re-evaluate it under the standard BEV protocol, or evaluate your own method under both.

---

## 14. Results reported (paper claims, not reproduced)

| Method | WT MODA | WT MODP | WT Prec | WT Rec | WT F1 | MVX MODA | MVX MODP | MVX F1 |
|---|---|---|---|---|---|---|---|---|
| MVDet | 88.2 | 75.7 | 94.7 | 93.6 | 94.1 | 83.9 | 79.6 | 91.5 |
| MVDeTr | 91.5 | **82.1** | **97.4** | 94.0 | 95.7 | 93.7 | **91.3** | 96.9 |
| MVAug | 93.2 | 79.8 | 96.3 | 97.0 | 96.6 | 95.3 | 89.7 | 97.6 |
| 3DROM | 93.5 | 75.9 | 97.2 | 96.2 | 96.7 | 95.0 | 84.9 | 97.5 |
| MVFP | 94.1 | 78.8 | 96.4 | 97.7 | 97.0 | 95.7 | 82.1 | 97.8 |
| TrackTacular | 93.2 | 77.5 | 97.3 | 95.8 | 96.5 | 96.5 | 75.0 | 98.2 |
| **CaMuViD** | **95.0** | 80.9 | 96.3 | **98.6** | **97.4** | **96.5** | 89.3 | **98.3** |

**Cross-dataset (train MultiviewX → test Wildtrack), [PAPER] Table 2:** CaMuViD **86.4 MODA / 93.5 F1** vs MVFP 76.7 / 88.8, GMVD 66.1 / 83.3, MVDet 17.0 / 54.0.

**Ablation, [PAPER] Table 4:** Sum, no FRM 93.8 → Concat, no FRM 94.6 → Concat + FRM **95.0**. Replacing `A^p/A^b` with camera-parameter projections: **−9.1 MODA**.

| Metric | Paper reported | My reproduction | Difference | Explanation |
|---|---|---|---|---|
| Wildtrack MODA | 95.0 | **not attempted** | — | No GPU/PyTorch, no images, no weights, DCNv3 needs `nvcc` (§11) |
| MultiviewX MODA | 96.5 | not attempted | — | " |
| Cross-dataset MODA | 86.4 | not attempted | — | " |
| **Camera-elimination recall (k=1..7)** | 60.2 … 98.6 | **oracle ceiling computed from shipped GT** (§12) | see §12.2 | ✅ **This one I could partially check without a model** |

**[INF] On the cross-dataset result (86.4 vs 76.7):** it is the paper's most impressive claim, and §7 makes it *more* surprising, not less — a model whose views are aligned by raster index should transfer poorly to a different rig. The paper notes it excludes one Wildtrack camera to keep N=6. **[INF] I cannot explain this result from the code, and I flag it as the claim most in need of independent verification.** Plausible partial explanations: (a) the per-view Cascade R-CNN heads are strong single-view detectors and the per-identity TP rule rewards "detected in ≥1 view"; (b) the InternImage/COCO backbone generalises well; (c) protocol differences (§13) compound across datasets. None of these is verifiable without running the code.

---

## 15. Controlled experiments — support in this repo

| Experiment | Supported? | Notes |
|---|---|---|
| **A. Number of views** | ❌ (as *retraining*) / ✅ (as *blanking*) | `num_head` fixes the architecture. Blanking (`blank_views`) is the only supported variant |
| **B. View selection** | ✅ | `Config_d.blank_views = ['C4','C5']` — any subset, no retraining. **[OBS]** Use my greedy order (C6→C3→C1→C7→C2→C4→C5) rather than index order for a much more informative curve |
| **C. View order** | ⚠ untested | Order-dependent by construction (concat + per-view heads). Permuting inputs without permuting heads should break it badly — a useful negative control |
| **D. Missing views** | ✅ **first-class** | This *is* `blank_views`. Note the GT-skip on L329–330 is commented out — decide deliberately whether to re-enable it |
| **E. View quality** | ✅ easy | `Config_d.transform` is a plain list of torchvision transforms |
| Fusion ablation (sum vs concat) | ⚠ code present but commented out | L707–712 contains the summation path, disabled |
| FRM ablation | ✅ | `refiningmodule_i` can be bypassed |
| `L_vp` weight ablation | ⚠ | Weight 1e−4 is hard-coded; **no ablation reported** |

---

## 16. Performance / cost **[INF]**, analytic from **[CODE]**

| Component | Params | Note |
|---|---|---|
| InternImage-T backbone | ~30 M | shared across views |
| FPN | ~3 M | shared |
| `projection_net_1` + `_2` | **2 × ≈16.8 M ≈ 33.6 M** | dominated by `Linear(256 → 65 536)` |
| `self.conv` (fusion) @ N=7 | 256·7·256 ≈ 0.46 M | 1×1 |
| `refiningmodule_i` × 7 | 7 × (256·256·49 + 256·256·9) ≈ **7 × 3.8 M ≈ 26.6 M** | 7×7 then 3×3 convs, **per view** |
| RPN + Cascade ROI heads × 7 | large (3 cascade stages each) | **per view** |

**[INF]** The paper reports no FLOPs, latency or memory in the tables — though the code prints per-stage timings (`"average timing for projection/fusion/refinement"`, L703/728/755) and `get_flops.py` exists. **[INF] The per-view head replication is the dominant cost and it scales linearly in N; this is the least efficient design in your set.**

**Answer to "accurate because of good multi-view strategy, or because it's big?"** **[INF] Substantially the latter, plus a favourable protocol.** The multi-view mechanism (§7) does no geometric work; the gains plausibly come from (i) a strong COCO-pretrained InternImage/DCNv3 backbone, (ii) N independent Cascade R-CNN heads, and (iii) an image-space per-identity metric (§13). The concat-vs-sum ablation (+0.8) and FRM (+0.4) are small next to those.

---

## 17. Failure analysis

| Failure type | Evidence | Probable cause | Potential solution |
|---|---|---|---|
| **7th camera hurts** | **[PAPER]** Table 3: 6 cams 95.6 → 7 cams 95.0 MODA, recall −0.7 | Redundant, narrow-FoV camera (**[OBS]** largest mean box area, 62 % coverage) adds duplicate/false evidence; concat cannot down-weight it | **Learned view gating** — the central gap |
| Single-view collapse | MODA 60.1 at k=1 vs **[OBS]** 91.9 % oracle → only 65.5 % of achievable | Severe occlusion in one view; nothing to corroborate against | Expected; establishes the value of multi-view |
| MODP lowest among top methods on Wildtrack (80.9 vs MVDeTr's 82.1) | **[PAPER]** Table 1 | IoU-based MODP is stricter (**[PAPER]** admits this) **and** no spatial alignment means box extents are refined per view only | — |
| Cross-dataset MODP collapses to 60.7 (worst in Table 2) | **[PAPER]** Table 2 | High recall (98.1) but poor localisation on an unseen rig; consistent with §7 (layout memorised per camera) | Real geometric alignment |
| **[OBS]** Cross-view scale ignored | Median **2.73×** height ratio for the same person across cameras; 56.5 % span ≥2 FPN levels | FPN levels are fused **independently** per level; no cross-scale, cross-view interaction | P2/MSMVD's BEV-FPN idea |
| Cannot handle a changed camera count | `num_head` hard-wired | Fixed-N architecture | N-agnostic pooling |
| Dead/commented code paths | L521–538, L707–712, L757–769 | Research code left in place | Reproducibility risk: unclear which path produced the published numbers |

---

## 18. Limitations

**Methodological** — no spatial alignment between views (§7); fixed N; order-dependent; no view weighting; per-view heads memorise static camera layouts; `L_vp` weight 1e−4 makes the invertibility constraint nearly vacuous; output is 2D boxes only (no BEV/3D).
**Dataset** — Wildtrack/MultiviewX only, 400 frames each, near-saturated; **[OBS]** massively over-covered (3–4 cameras already give ≈100 % coverage), so these benchmarks cannot separate coverage from disambiguation without the analysis in §12.
**Evaluation** — image-space IoU + per-identity TP, different from every tabulated competitor (§13).
**Computational** — 33.6 M params in two projection FCNs alone; N replicated Cascade R-CNN heads; no FLOPs/latency reported; two-stage training (backbone finetune → fusion).
**Generalisation** — the strong cross-dataset number is unverifiable from code and hard to reconcile with §7.
**Implementation** — needs custom DCNv3 CUDA op; pinned mmcv 1.5.0/mmdet 2.28.1; hard-coded paths and `num_head`; substantial commented-out code; `print()` debugging left in the forward pass (L703, L728, L755) including a hard-coded `/7`.

---

## 19. Research gaps

**Gap 1 — Calibration-free multi-view fusion still lacks any *spatial* correspondence mechanism.**
CaMuViD shows you can drop calibration and still win on these benchmarks, but **[CODE]** its "projection" is spatially uniform channel mixing (§7). The genuinely open problem — *learning cross-view spatial correspondence without calibration* — remains unsolved.
> **Direction:** learn correspondence with **cross-view attention over spatial tokens** (P5's mechanism, minus the camera-aware embedding) so that view *i*'s query attends to *geometrically relevant* locations in view *j*. This is calibration-free **and** spatially aligned — a combination none of your six papers has.

**Gap 2 — Redundant views degrade performance, and no method can detect or suppress this.**
**[PAPER]**'s own Table 3: 7 cameras < 6 cameras. **[OBS]**: C4/C5 add 0.00 pp coverage; C2/C4/C5 are jointly redundant; C1↔C6 Jaccard = 92.8 %.
> **Direction:** a **view-informativeness estimator** that scores each view per frame (or per BEV cell) and gates its contribution. Success criterion is concrete and pre-registered: **7-camera performance ≥ the 6-camera result (95.6 MODA)**, i.e. adding a view must never hurt.

**Gap 3 — "More views help" conflates coverage with disambiguation.**
**[OBS]** §12 separates them for the first time here: C4/C5 give +4.8 pp recall at **zero** coverage gain.
> **Direction:** report both curves — *oracle coverage* and *achieved recall* — for every view-count experiment. Then design fusion that explicitly models **agreement/corroboration** (e.g. count and variance of supporting views), not just feature aggregation.

**Gap 4 — Missing-view robustness is evaluated by blanking, never by retraining.**
**[CODE]** `blank_views` feeds black images to a 7-view model. A black image is an *out-of-distribution input*, not an absent one.
> **Direction:** compare three regimes — (a) blank the view, (b) mask it out of an N-agnostic pooling, (c) retrain at that N. **[INF] Only (b) and (c) measure "fewer views"; (a) measures "corrupted views".** No paper in your set makes this distinction.

**Gap 5 — Protocol fragmentation blocks fair comparison.**
> **Direction:** publish a **single evaluation harness** applying both the BEV-distance and image-IoU protocols to all baselines. Modest, but genuinely useful to the field, and it protects your own results from this criticism.

---

## 20. What to borrow

| Useful idea | Why it works | Limitation | How to adapt |
|---|---|---|---|
| **Detect in image space, not only BEV** | Preserves appearance detail that ground-plane warping destroys; avoids shadow/stretch artefacts for distant targets | Loses a unified scene-level representation; needs per-view heads | Consider a **dual-output** head: BEV occupancy *and* per-view boxes, sharing a fused representation |
| **Drop the calibration dependence** | Removes the main obstacle to deploying on a new rig; motivates the strong cross-dataset claim | As implemented, achieved by removing alignment altogether | **Keep the goal, change the mechanism** — learn correspondence via attention (Gap 1) |
| **Concat rather than sum**, with a stated reason | Summation averages a confident view with an uncertain one; concat lets the network decide. Worth +0.8 MODA in their ablation | Order-dependent, O(N) params | **[INF] Their argument is really an argument for *weighting*, not concatenation.** A confidence-weighted sum gets the benefit while staying permutation-invariant — this is a clean, testable hypothesis for your work |
| **Cycle-consistency between projection and back-projection** | An elegant, parameter-free way to regularise a learned transform | Weight 1e−4 ⇒ near-vacuous; never ablated | Ablate the weight properly; a meaningful setting could make it a genuine constraint |
| **Feature Refinement Module (large 7×7 kernel) after fusion** | Restores spatial context that 1×1 fusion cannot provide. Worth +0.4 MODA and +0.7 MODP | Per-view, so ×N cost | Echoes MVDet's large-kernel finding — **large receptive fields after fusion are consistently valuable across papers** |
| **Shipping the annotation JSONs** | Let me run three original analyses **without any GPU, dataset or model** (§10, §12) | — | **Copy this practice.** Shipping labels (not just code) is what made this the most analysable folder in your set |

---

## 21. Verdict

CaMuViD is **the most interesting paper in your set to critique, and the least suitable to adopt.**

- **Do not** make it your baseline: its evaluation protocol is not comparable to the standard BEV one (§13), and the released code does not implement the geometric alignment its framing implies (§7).
- **Do** mine it for two things: the **calibration-free objective** (a real, valuable goal, unsolved) and — most importantly — its **camera-elimination table**, which after my cross-check against the data (§12) becomes the strongest quantitative evidence available anywhere in your six papers that **redundant views can actively hurt**, and that **coverage and disambiguation are separable contributions**.
- **Do** reuse the shipped annotations: they enabled every original measurement in this report set.

**Cross-references:** P1 (MVDet) is the calibrated counterpart it argues against · P2 (MSMVD) addresses the cross-view scale problem it ignores · P5 (CVT) provides the learned, calibration-aware attention that would fix Gap 1.
