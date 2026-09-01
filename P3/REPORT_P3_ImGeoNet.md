# P3 — ImGeoNet: Image-induced Geometry-aware Voxel Representation for Multi-view 3D Object Detection

**Analysis date:** 2026-08-29
**Evidence base:** paper PDF (ICCV 2023, 10 pp.) + full source (`ImGeoNet/`) + **three genuine training logs (932 KB)**
**Execution status:** ❌ not executed here (§11) — **but this folder contains the only real experimental records in your entire set**, and I mined them extensively.

Tags: **[PAPER]** claimed · **[CODE]** verified in source · **[LOG]** read from the shipped training logs · **[OBS]** measured by me · **[INF]** my inference

---

## 1. Inventory

| Item | Value |
|---|---|
| Title | ImGeoNet: Image-induced Geometry-aware Voxel Representation for Multi-view 3D Object Detection |
| Authors | Tao Tu¹, Shun-Po Chuang², Yu-Lun Liu³, Cheng Sun¹, Ke Zhang⁴, Donna Roy⁴, Cheng-Hao Kuo⁴, Min Sun¹˒⁴ (¹NTHU ²NTU ³NYCU ⁴Amazon) |
| Venue / Year | **ICCV 2023** (pp. 6996–7005) |
| Code | `ImGeoNet/` — full mmdetection3d 0.8.0 fork |
| README | Yes, with a performance table linking the three logs |
| Datasets | ScanNetV2, ScanNet200, ARKitScenes — **none present**; only split lists + preprocessing scripts |
| Pretrained weights | **None provided anywhere** (no link, no file) |
| Configs | `configs/imgeonet/{imgeonet,imvoxelnet}_{scannet,scannet200_vx808032,arkit}.py` — **6 configs: 3 method + 3 matched baselines** |
| Train / test scripts | `script/3_train_imgeonet.sh`; `tools/test.py $config $ckpt --eval mAP` |
| Requirements | `script/0_install_env.sh` — torch 1.7.1+cu110, mmcv-full 1.2.7, mmdet 2.10.0, numpy 1.23.0 |
| **Result files** | ✅ **`logs/scannet.txt` (130 KB), `logs/arkit.txt` (174 KB), `logs/scannet200.txt` (627 KB)** — full mmdet training logs with env, config, per-iteration losses and per-epoch per-class evaluation |
| Reproducibility | **Blocked here** (no GPU/data/weights), but the **evidence quality is the best of the six** |

---

## 2. Relevance to *Multi-View Object Recognition*

**A. Problem solved.** Detect and **classify** 3D objects (18 / 189 / 17 categories) in an indoor scene from an arbitrary number of posed RGB images — **without depth or point clouds at inference**.

**B. What "multi-view" means here.** **Many viewpoints of the same scene and the same objects**, captured by a *moving* hand-held camera (iPad Pro / RGB-D scanner). *N* is a free variable: **20 views at training, 50 at test**, and the paper sweeps **10 → 100** views. Views are unordered and the model is permutation-invariant.

**C. Input.** `{I_t} ⊆ ℝ^{H×W×3}` with intrinsics `{K_t} ⊆ ℝ^{3×3}` and poses `{T_t} ⊆ SE(3)`. ScanNet 640×480 (padded), ARKit 192×256.
**[CODE]** Depth maps *are* loaded — but **only in the training pipeline** (`Collect3D` keys include `depth_maps`, `depth_masks`), used solely to supervise the occupancy head. The ARKit test pipeline collects `keys=['img']` only. **The paper's "only images at inference" claim is verified in code.** ✔

**D. Output.** Per-object **category label + axis-aligned/oriented 3D box** `b ∈ ℝ⁷ = (x,y,z,w,h,l,φ)`.

**E. Relevance score.**

| Reading | Score | Why |
|---|---|---|
| *Multi-viewpoint recognition of objects* | **4.5 / 5** | This is the closest thing in your set to genuine multi-view **object recognition**: many unordered viewpoints of the *same objects*, variable N, and a **category** output. It even reports the exact **views-vs-accuracy curve** your Experiment A asks for |
| *Multi-camera scene detection* | **3.5 / 5** | Different sensor setup (moving camera vs fixed rig), same fusion questions |

**Net: 4 / 5 — the highest in your set.** ✅ **My recommended primary baseline for the "many viewpoints" branch of your work.**

---

## 3. Research problem

- **Input:** `{I_t, K_t, T_t}`, arbitrary count.
- **Output:** `{b} ⊆ ℝ⁷` + category.
- **Assumptions:** known intrinsics and poses (SLAM/ARKit-provided); boxes on the ground → only yaw predicted; **depth available at *training* time only**.
- **Constraints:** feature volume fixed at 6.4 × 6.4 × 2.56 m (40×40×16 @ 0.16 m; 80×80×32 @ 0.08 m for ScanNet200).
- **Objective:** maximise mAP@0.25 / mAP@0.50.

---

## 4. Motivation and the exact gap

**[PAPER]** ImVoxelNet (the prior SOTA) builds a voxel feature volume by back-projecting 2D features **along the whole camera ray**: every voxel on a ray gets the *same* feature, whether it lies on a surface or in empty space. The volume is therefore **geometry-unaware** — free-space voxels carry object-like features and generate false positives (Fig. 1, cyan "missed free-space voxels").

**The gap:** *feature-volume methods do not know where the surfaces are.* Naively fixing this needs MVS depth estimation, which is expensive (the paper's own MaGNet baseline: **15.7× the runtime**, 42 % larger model, 5× more input views).

**ImGeoNet's answer:** predict a per-voxel **surface probability** `S` with a small 3D encoder-decoder *reusing the same feature volume*, and multiply: `V_g = S ⊙ V`. Depth supervises `S` during training only.

---

## 5. Previous approaches **[PAPER]** §2

| Category | Examples | What it does | Limitation |
|---|---|---|---|
| Point-cloud detection | VoteNet, H3DNet, FCAF3D, GroupFree3D | Hough voting / sparse 3D convs on points | Needs expensive depth sensors; struggles on sparse/noisy scans (ARKit) and small objects (downsampling drops them) |
| Monocular 3D | ImVoteNet, Total3D, PerspectiveNet | Single image → 3D | Scale ambiguity, occlusion, limited FoV |
| Depth-assisted image 3D | pseudo-LiDAR, CaDDN, MaGNet-cascade | Estimate depth, back-project | Extra backbone; **15.7× runtime** for the cascade baseline |
| **Multi-view feature volume** | Atlas, **ImVoxelNet** | Back-project 2D features into a voxel grid, average | **Geometry-unaware** — free space contaminated (the gap) |
| Multi-view BEV (driving) | BEVDet, BEVFormer, DETR3D, PETR | BEV representation for outdoor | Assumes on-ground objects and a fixed surround rig; poor fit for cluttered indoor scenes |

---

## 6. Proposed method

```
 I₁ ─┐
 I₂ ─┤  ResNet-50 (ImageNet)  →  FPN(256)   [shared weights, all views]
 ... ┼──────────────► F_t ∈ ℝ^{H×W×256}, stride 4  (asserted in code)
 I_T ─┘                        │
                     back-project along ray (Atlas-style):
                     V_t[x,y,z] = F_t[u,v],  M_t = in-frustum mask
                               │
        ┌──────────────────────┴───────────────────────┐
        │  MEAN over views     V   = ΣV_t⊙M_t / ΣM_t   │   (Eq. 4)
        │  VARIANCE over views V_var = E[V²] − E[V]²   │   (Eq. 8)
        └──────────────────────┬───────────────────────┘
                    V' = concat(V, V_var)  ∈ ℝ^{512}
                               │
                  ┌────────────▼─────────────┐
                  │ OccupancyHead g(·)       │  3D Enc-Dec + residual
                  │ → Linear → Sigmoid       │  supervised by FocalLoss(w=10)
                  └────────────┬─────────────┘  against depth-derived surface voxels
                               │  S ∈ [0,1]
                       V_g = S ⊙ V              (Eq. 6)   ← note: weights the MEAN only
                               │
                 FastIndoorImVoxelNeck (multi-scale 3D convs, L=3)
                               │
                 ScanNet/SunRgbd ImVoxelHeadV2 (anchor-free, FCOS-style)
                               ▼
              class + centerness + 3D box, NMS (iou 0.25, score 0.01)
```

### Module detail **[CODE]** `mmdet3d/models/detectors/imgeonet.py`

| Module | In | Out | Operation | Line |
|---|---|---|---|---|
| `backbone` + `neck` | `T×3×480×640` | `T×256×120×160` | ResNet-50 (frozen stage 1, `norm_eval=True`) + FPN; **`assert stride == 4`** | 70–76 |
| `backproject` | features, voxel centres | `V_t`, `valid_t` | nearest-neighbour lookup `F_t[y,x]` per voxel per view; out-of-frustum → 0 | 284–303 |
| **mean** | `{V_t}` | `V` | `vol.sum(0) / valid_count`; invalid voxels zeroed | 103–107 |
| **variance** | `{V_t}` | `V_var` | `E[X²] − E[X]²` across views | 110 |
| `occ_head` | `concat(V, V_var)` = 512 ch | `S` | 3D Enc-Dec, Linear, Sigmoid | 134/160 |
| geometry shaping | `S`, `V` | `V_g` | `avg_vols * occ` | **142 / 167** |
| `neck_3d` | `V_g` | multi-scale volumes | `FastIndoorImVoxelNeck(256→128, n_blocks=[1,1,1])` | 122 |
| `bbox_head` | volumes | boxes | ImVoxelNet head, L=3 scales, `limit=27`, `centerness_topk=18` | — |
| `compute_target_occ` | depth maps + poses | binary surface volume | voxel is positive if its depth matches GT depth within `margin = voxel_size_z · depth_cast_margin/2 = 0.32 m` | 245–281 |

**[INF] A subtle and important design fact:** the variance volume is used **only** as *input to the occupancy head* (line 135/161). The detector consumes `avg_vols * occ` — the **mean**, gated. So cross-view disagreement informs *where the surfaces are*, but never reaches the classifier directly. **That is an exploitable gap (§18, Gap 2).**

---

## 7. Multi-view fusion strategy — the key section for you

**Mechanism: masked MEAN pooling + a VARIANCE side-channel, then multiplicative geometric gating.**

```python
valid_count = valid.sum(dim=0)                  # how many views see this voxel
avg_vol  = volume.sum(0) / valid_count          # Eq. 4  — MEAN, normalised by valid views
avg_vol2 = volume.pow(2).sum(0) / valid_count
var_vol  = avg_vol2 - avg_vol.pow(2)            # Eq. 8  — VARIANCE
avg_vol[:, invalid_voxel_mask] = 0.
x = self.extract_feat(avg_vols * occ)           # Eq. 6  — gate the MEAN by surface prob
```

**Why this is the best-engineered fusion in your set — [INF]:**

1. **Permutation-invariant.** Mean and variance are symmetric functions of the view set. Camera order is irrelevant, by construction.
2. **N-agnostic — and they exploit it.** **[CODE][LOG]** The model **trains with `n_images=20` and tests with `n_images=50`**, in all three configs. This train/test view-count mismatch is only possible because fusion is a normalised statistic. MVDet (P1) and CaMuViD (P4) cannot do this at all.
3. **Explicit validity normalisation.** Dividing by `valid_count` (not by *N*) means a voxel seen by 3 views is not artificially attenuated relative to one seen by 40. **[INF] This is exactly the fix MVDet lacks**, where out-of-FoV regions are zero-padded and indistinguishable from "empty".
4. **Variance = a free multi-view consistency cue.** Photometric/feature agreement across views is the classical MVS surface signal. Reusing the detector's own features to get it — instead of running a depth network — is the paper's cleverest move: **[PAPER]** the geometry-shaping head costs only **+53.6 ms** (139.0 → 181.8 ms at 50 views, ~24 % relative) and **+18 %** model size, versus **15.7×** runtime for the MaGNet cascade.
5. **Multiplicative gating is a soft, spatial view-agnostic mask.** It suppresses free-space voxels without a hard threshold.

**Weaknesses — [INF]:**

1. **The mean is still uniform across views.** A view where the object fills the frame and a view where it is 8 px in a corner contribute equally. There is **no per-view weighting anywhere in the model.**
2. **Variance never reaches the detector** (see §6 note). Disagreement is consumed by the occupancy head and then discarded.
3. **`S` is view-independent.** One scalar per voxel for the whole scene. It cannot express "this voxel is a surface, but view 7 is looking at an occluder".
4. **Nearest-neighbour back-projection** (`.round().long()`, line 293–294) — no bilinear sampling, so sub-voxel accuracy is quantised. **[INF] Plausibly a contributor to the AP@0.50 collapse in §16.**
5. **Fixed 6.4 × 6.4 × 2.56 m volume**; larger rooms need `RandomShiftOrigin` at train and relocation at test.
6. **Requires depth at training time.** The method is "image-only at inference", not "image-only".

---

## 8. Loss function **[PAPER]** §4, **[CODE]** configs

```
L = L_cls(Focal, γ=2, α=0.25)          weight 1
  + L_centerness(CrossEntropy)          weight 1
  + L_bbox(Rotated/AxisAligned 3D IoU)  weight 1
  + L_occ(Focal, γ=2, α=0.25)           weight 10      ← the only re-weighted term
```
**[PAPER]** *"All the loss weights are set to 1, except for the surface voxel prediction loss, which is assigned a weight of 10... we employed identical loss hyperparameters across all experiments."* **[CODE]** `loss_weight=10.0` in `occ_head` of all three configs. ✔ Verified.

**[INF] Why 10×:** surface voxels are a tiny minority of the 40×40×16 = 25 600-voxel grid, and the occupancy signal must be strong enough to survive being multiplied into features that also receive detection gradients. **[LOG]** In the ScanNet log, `loss_occ ≈ 0.36` and `acc_occ ≈ 0.93` at convergence — the occupancy head is easy to fit and stable.

---

## 9. Datasets

| | ScanNetV2 | ScanNet200 | ARKitScenes |
|---|---|---|---|
| Classes | 18 | **189 evaluated** (200 defined) | 17 |
| Scenes | 1201 train / 312 val | same split | 4498 / 549 captures (1661 scenes) |
| Image res | 968×1296 → resized 640×480, padded | same | 192×256 |
| Views: train / test | **20 / 50** | 20 / 50 | 20 / 50 |
| Sampling | train `random`, test `linear` (uniform over the trajectory) **[CODE]** `multi_view.py:21-25` | same | same |
| Voxel | 0.16 m, 40×40×16 | **0.08 m, 80×80×32** (for small objects) | 0.16 m, 40×40×16 |
| Boxes | axis-aligned (from semantic labels, VoteNet protocol) | same | oriented |
| `RepeatDataset times` | 3 | 3 | 3 |
| Epochs / LR steps | 12, step [8,11] | **30, step [8,29]** | 12, step [8,11] |
| `samples_per_gpu` | 1 | 1 | **8** |
| Augmentation | `RandomShiftOrigin(std=(.7,.7,0))` **only** | same | same |
| Depth | needed at train only | same | low-res, noisy (paper notes ARKit depth is inferior) |

**[INF] Is this representative of your problem?** **More so than any other paper here.** You get: many unordered viewpoints of the same object, variable N, real category labels, a long-tail regime (ScanNet200), and a *realistic-capture* regime (ARKit). The main mismatch is that objects are embedded in cluttered scenes rather than isolated.

---

## 10. Experimental setup — **[LOG]** (this is real, recorded data)

| | `scannet.txt` | `arkit.txt` | `scannet200.txt` |
|---|---|---|---|
| Date | **2023-02-01** | 2024-08-25 | 2024-08-25 |
| GPU | **4 × Tesla P40** | **1 × RTX 3090** | **3 × RTX 3090** |
| Distributed | True | False | False |
| Python / PyTorch | 3.8.16 / **1.7.1+cu110** | same | same |
| MMCV / MMDet / MMDet3D | 1.2.7 / 2.10.0 / **0.8.0+a75dd1a** | 1.2.7 / 2.10.0 / **0.8.0+a5dfdf3** | same as arkit |
| CUDA runtime / NVCC | 11.0 / 10.0.130 | " | " |
| `model.type` in config | ⚠ **`ImVoxelNet`** (with `occ_head`) | `ImGeoNet` | `ImGeoNet` |
| Peak GPU memory | 10 776 MB | **19 417 MB** | 15 236 MB |
| Iter time | 1.76 s | 2.66 s | 1.39 s |
| Iters/epoch | 901 | 1685 | 3603 |
| Epochs completed | **12 / 12 ✔** | **12 / 12 ✔** | ⚠ **11 / 30** |
| Wall-clock | ~10:24 → 16:16 (≈ 5 h 52 m) | ~13 h | ~19 h, then **stopped** |

**Optimiser (all three):** AdamW, lr 1e−4, wd 1e−4, `backbone lr_mult=0.1`, grad-clip max_norm 35.

### ⚠ Three discrepancies I found in the logs

1. **`scannet.txt` declares `type='ImVoxelNet'`, not `ImGeoNet`** — yet its loss stream contains `loss_occ` and `acc_occ`, and the released `imvoxelnet_scannet.py` baseline has **no** `occ_head`. **[INF]** This log predates the class rename; functionally it *is* ImGeoNet. Different git hash (`a75dd1a` vs `a5dfdf3`) and a 18-month gap support that. Not a correctness problem, but it means the ScanNet log was produced by a **different code revision** than the one you have.
2. **ScanNet200 training was abandoned at epoch 11 of 30.** The log's last entries are `Saving checkpoint at 11 epochs`, then the final evaluation table, then nothing — with `eta: 1 day, 4:45:03` remaining. **The second LR drop (`step=[8,29]`) never happened.** The README's reported 22.38 / 9.67 is exactly this epoch-11 table. **[INF] The published ScanNet200 number therefore comes from a ~37 %-complete training schedule.**
3. **Hardware differs per dataset** (4×P40 / 1×3090 / 3×3090), and the README itself warns *"Performance may vary slightly depending on the number of GPUs."*

---

## 11. Attempt to execute — failed, documented

```
OS      Windows 11        Python 3.14.3 (conda base only)
PyTorch NOT INSTALLED     (ModuleNotFoundError: No module named 'torch')
GPU     NONE              (nvidia-smi not found)
```

| Problem | Cause | Solution attempted | Result |
|---|---|---|---|
| No PyTorch | Absent from the only env | `conda env list`, `import torch` | ❌ |
| No CUDA GPU | Machine has no NVIDIA device | `nvidia-smi` | ❌ |
| Needs torch **1.7.1+cu110** | mmcv-full 1.2.7 has no wheel for Python 3.14; cu110 wheels stop at Python 3.9 | Read `script/0_install_env.sh` | ❌ Would need a Python 3.8 env + CUDA 11.0 toolchain |
| Custom CUDA op | `mmdet3d/ops/rotated_iou/cuda_op` compiled via `python setup.py install` | Read install script | ❌ Requires `nvcc` |
| **mmcv 1.2.7 is Linux-first** | Windows wheels for that version are not published on the openmmlab index used in the script | Read install script | ❌ Linux effectively required |
| No dataset | ScanNet frame extraction is explicitly warned to need *"a lot of disk space"*; ARKit needs `2a_download_arkit.sh` | Listed `data/` — only split lists + scripts | ❌ |
| No pretrained weights | None linked, none on disk | Searched tree for `*.pth` | ❌ **Zero checkpoints anywhere** |
| Even `tools/test.py` impossible | Needs both a checkpoint and the dataset | — | ❌ |

**Nothing was run. Every ImGeoNet number below is either [PAPER] or [LOG].**

---

## 12. Results: paper vs the shipped logs — the most valuable comparison available

**[LOG] Per-epoch validation curve, extracted by me** (`research_analysis/04_results/`):

| Epoch | ScanNet mAP@.25/.50 | ARKit mAP@.25/.50 | ScanNet200 mAP@.25/.50 |
|---|---|---|---|
| 1 | 30.01 / 9.97 | 38.76 / 14.90 | 6.61 / 2.00 |
| 2 | 38.91 / 14.27 | 47.03 / 20.89 | 12.17 / 3.85 |
| 3 | 41.47 / 16.45 | 52.17 / 28.79 | 14.62 / 5.70 |
| 4 | 43.55 / 19.11 | 53.93 / 34.50 | 15.59 / 6.56 |
| 5 | 48.68 / 22.86 | 54.34 / 34.86 | 17.28 / 6.92 |
| 6 | 49.06 / 22.92 | 56.09 / 36.35 | 19.46 / 7.45 |
| 7 | 51.10 / 24.96 | 57.00 / 38.95 | 20.12 / 7.70 |
| 8 | 50.84 / 24.90 | 57.28 / 39.05 | 18.14 / 6.97 |
| **9 (LR drop)** | **54.52 / 28.40** | **59.18 / 42.82** | **22.50 / 9.50** |
| 10 | 54.69 / 28.87 | 59.48 / 42.48 | 21.72 / 9.42 |
| 11 | 54.46 / 28.20 | 59.94 / 42.44 | **22.38 / 9.67** ← reported, run ends |
| 12 | **54.57 / 28.94** ← reported | **59.82 / 42.76** ← reported | (never ran) |

### Paper vs log vs README

| Dataset | Metric | **Paper** | **README** | **Log (final)** | Δ paper−log | Assessment |
|---|---|---|---|---|---|---|
| ScanNetV2 | mAP@0.25 | 54.8 | 54.57 | **54.57** (ep 12) | **−0.23** | README = log exactly ✔ |
| ScanNetV2 | mAP@0.50 | 28.4 | 28.94 | **28.94** (ep 12) | **+0.54** | Paper's 28.4 = **epoch 9's** value (28.40) |
| ARKitScenes | mAP@0.25 | 60.2 | 59.82 | **59.82** (ep 12) | −0.38 | README = log exactly ✔ |
| ARKitScenes | mAP@0.50 | 43.4 | 42.76 | **42.76** (ep 12) | −0.64 | Best epoch was 9 (42.82) |
| ScanNet200 | mAP@0.25 | 22.3 | 22.38 | **22.38** (ep 11 of 30) | +0.08 | ⚠ **incomplete run** |
| ScanNet200 | mAP@0.50 | — | 9.67 | 9.67 | — | " |

**[INF] Interpretation — this is a well-behaved, honest release.**
- README numbers match the logs to the second decimal: the authors published the *actual* runs, not cherry-picked bests.
- Paper numbers are **0.2–0.6 points higher** than the shipped logs on 3 of 4 comparable metrics. That is exactly the size the README's own caveat predicts (*"Performance may vary slightly depending on the number of GPUs"*), and the logs used 4×P40 / 1×3090 / 3×3090 — none of which need be the paper's configuration.
- ⚠ **The single caveat:** ScanNetV2 mAP@0.50 in the paper (28.4) coincides precisely with epoch 9, while epochs 10 and 12 are *higher* (28.87, 28.94). The README/logs are therefore *more* conservative than the paper here, not less.
- ⚠ **ScanNet200 is the weak point:** the published 22.38 comes from an 11/30-epoch checkpoint whose LR schedule never completed. The trend (ep 9: 22.50, ep 10: 21.72, ep 11: 22.38) is still oscillating. **[INF] The true converged number is unknown and could plausibly be higher.** If you cite this number, say so.

| Metric | Paper reported | **My reproduction** | Difference | Explanation |
|---|---|---|---|---|
| any | see above | **not attempted** | — | No GPU, no PyTorch, no dataset, no weights (§11) |

---

## 13. Controlled experiments — ImGeoNet is the *only* repo here that supports them natively

| Experiment | Supported? | How |
|---|---|---|
| **A. Number of views** | ✅ **fully** | Change `n_images` in the **test** pipeline. No retraining, no architecture change — mean-pooling is N-agnostic. **[PAPER] Table 4 already does this (10→100).** |
| **B. View selection** | ✅ easy | `MultiViewPipeline.__call__` chooses `ids` (`random` or `linspace`). Swapping in a custom selector is a ~5-line change |
| **C. View order** | ✅ **provably irrelevant** | `sorted(ids.tolist())` + mean/variance ⇒ mathematically permutation-invariant. **A good control: it should give bit-identical results** |
| **D. Missing views** | ✅ | Just reduce `n_images`; `valid_count` renormalises automatically |
| **E. View quality** | ✅ | Insert corruption transforms into the `MultiViewPipeline.transforms` list |
| **Ablation: geometry shaping** | ✅ **shipped** | `configs/imgeonet/imvoxelnet_*.py` is the *identical* config minus `occ_head`, depth loading and `depth_cast_margin` — **verified by diffing the two files.** A perfect matched control |
| **Ablation: GT occupancy upper bound** | ✅ **shipped** | `use_gt_occ=True` in `ImGeoNet.__init__` replaces `S` with the true surface volume — this is the paper's Table 6 row (g-4) |

### **[PAPER] Table 4 — the views-vs-accuracy curve you need (ARKitScenes, mAP@0.25/@0.50)**

| Method | 10 | 20 | 30 | 40 | 50 | 75 | 100 |
|---|---|---|---|---|---|---|---|
| **ImGeoNet** | 39.0/21.9 | 53.1/34.3 | 57.1/39.2 | 59.5/42.7 | **60.2/43.4** | 61.8/45.0 | **62.4/45.7** |
| ImVoxelNet | 36.2/19.6 | 50.5/30.6 | 54.6/35.2 | 57.4/37.9 | 58.0/38.8 | 58.8/40.5 | 59.7/42.0 |
| VoteNet (points) | 30.2/20.8 | 45.9/31.5 | 50.2/34.0 | 51.1/36.8 | 53.3/38.5 | 53.6/38.3 | 53.9/39.0 |

**[INF] What this curve actually says — three findings that matter for your research:**

1. **Sharp saturation.** 10→20 views: **+14.1** mAP. 20→30: +4.0. 30→40: +2.4. 40→50: +0.7. 50→100: **+2.2 over a doubling and a half**. The marginal value of a view collapses by ~20× between the 2nd and 10th decade of views.
2. **Geometry shaping is worth ~2–3× the views.** ImGeoNet @ 40 views (59.5/42.7) ≈ ImVoxelNet @ 100 views (59.7/42.0). **[PAPER]** confirms this framing. So *representation quality substitutes for view count* — a strong, quantified statement.
3. **The gap is widest where views are scarcest.** At 10 views ImGeoNet leads ImVoxelNet by +2.8; at 100 views by +2.7 — roughly constant in absolute terms but **+7.7 % vs +4.5 % relative**. **[INF] Geometry priors matter most in the low-view regime — exactly the regime where adaptive view selection would also pay off.**

**[PAPER] Table 6 — geometry-shaping ablation (ScanNetV2), with cost:**

| Variant | rel. size | rel. runtime | mAP@.25 | mAP@.50 |
|---|---|---|---|---|
| ImGeoNet | 1.00 | 1.00 | **54.8** | **28.4** |
| ImVoxelNet | 0.82 | 0.85 | 48.7 | 23.8 |
| ImVoxelNet + MaGNet depth | 1.42 | **15.72** | 53.8 | 28.2 |
| ImVoxelNet + **GT depth** (upper bound) | 0.82 | 0.89 | **58.8** | **33.4** |

**[INF] The most useful number in this table is the last row.** The oracle-depth ceiling is 58.8/33.4 vs ImGeoNet's 54.8/28.4 — a **4.0 / 5.0 point gap that geometry shaping does not close**. The paper says so explicitly. On ARKitScenes the gap is smaller (2.0/3.0), because ARKit's "GT" depth is itself noisy. **That residual gap is an open research target.**

---

## 14. Performance / cost **[PAPER]** Table 5 + **[LOG]**

| | 20 views | 40 | 50 | 100 |
|---|---|---|---|---|
| ImGeoNet inference (ms) | 139.0 | 166.1 | 181.8 | 245.8 |
| ImVoxelNet (ms) | 113.1 | 140.0 | 155.9 | 219.7 |
| **Geometry-shaping overhead** | +25.9 | +26.1 | **+25.9** | +26.1 |

**[INF] The overhead is a constant ~26 ms**, independent of view count — because `g(·)` runs once on the *fused* volume, not per view. That is an excellent scaling property and the strongest engineering argument for the design. Relative cost falls from 23 % (20 views) to 12 % (100 views).

Model size: **485.6 MB** (ImGeoNet, ScanNetV2), inference **489.9 ms** on a single RTX 3090 including data loading. **[LOG]** Training memory 10.8–19.4 GB.

**Answer to "accurate because of a good multi-view strategy, or because it's bigger?"**
**[INF] Genuinely the strategy, and the paper proves it properly.** ImGeoNet is *18 % larger and 18 % slower* than ImVoxelNet yet gains **+6.1 / +4.6 mAP**; the MaGNet variant is *42 % larger and 1472 % slower* and gains **less** (+5.1/+4.4). Cost-normalised, geometry shaping dominates.

---

## 15. Ablations present in the repository

| Ablation | Available? | Mechanism |
|---|---|---|
| w/o geometry shaping | ✅ | `configs/imgeonet/imvoxelnet_*.py` (verified matched diff: only `occ_head`, depth loading, `depth_cast_margin` differ) |
| GT-occupancy upper bound | ✅ | `use_gt_occ=True` |
| Voxel resolution | ✅ | `voxel_size` / `n_voxels` (0.16 m vs 0.08 m configs both shipped) |
| Number of views | ✅ | `n_images` |
| Variance channel on/off | ⚠ code edit | `occ_head.in_channels = 256+256`; would become 256 and drop `var_vols` from the `cat` |

---

## 16. Failure analysis — **[LOG]**, from the shipped per-class evaluation tables

I parsed the final per-class table from each log (`research_analysis/09_scripts/log_class_stats.py`, output in `05_failure_analysis/E4_*`).

### ScanNetV2 (18 classes, mAP@.25 = 54.57)

| Worst 6 | AP@.25 | AR@.25 | AP@.50 |
|---|---|---|---|
| **picture** | **0.0427** | 0.1351 | 0.0025 |
| window | 0.2466 | 0.5213 | 0.0294 |
| curtain | 0.3372 | 0.6269 | 0.0426 |
| garbagebin | 0.3605 | 0.5736 | 0.1737 |
| door | 0.4041 | 0.6510 | 0.0804 |
| counter | 0.4117 | 0.7308 | 0.0786 |

| Best 4 | AP@.25 | AP@.50 |
|---|---|---|
| toilet | 0.9518 | 0.6883 |
| bed | 0.8407 | 0.7476 |
| bathtub | 0.8147 | 0.6165 |
| sofa | 0.7558 | 0.4566 |

### ARKitScenes (17 classes, mAP@.25 = 59.82)
`tv_monitor` **0.0355 / AP@.50 = 0.0000**; `stove` 0.2360 / 0.0228; `dishwasher` **AR@.25 = 0.88 but AP@.25 = 0.20**; best: bathtub 0.9456, toilet 0.9092, bed 0.8865.

### ScanNet200 (189 classes, mAP@.25 = 22.38)
- **63 classes (33.3 %) have AP@0.25 = 0.0000 *and* AR@0.25 = 0.0000** — never recalled once.
- **89 classes (47.1 %) have AP@0.50 = 0.**
- Median class AP@0.25 = **0.10** vs mean 0.2238 → heavily skewed.
- ⚠ Degenerate classes inflate the mean: `scale` and `guitar` both score **AP@0.25 = 1.0000 with AP@0.50 = 0.0000** — near-certainly 1–2 instances each.

### Failure table

| Failure type | Example (with evidence) | Probable cause | Potential solution |
|---|---|---|---|
| **Thin / flat / wall-parallel objects** | picture 0.043, window 0.247, curtain 0.337, door 0.404, tv_monitor 0.036 | These are **thinner than one 0.16 m voxel** and lie flush against a wall. The surface probability `S` cannot separate "picture" from "wall"; the mean-pooled feature averages both | Anisotropic / finer voxels near surfaces; predict a signed distance instead of binary occupancy; per-view feature retention |
| **Localisation collapse @ IoU 0.5** | showercurtain 0.475→0.020 (**−95.8 %**); curtain −87.4 %; counter −80.9 %; door −80.1 %. Mean AP drop across all 18 classes = **47.0 %** | Found approximately, sized wrongly. **[CODE]** back-projection uses `.round().long()` (nearest neighbour) → extents quantised to the voxel grid | Bilinear/trilinear back-projection; box refinement; higher-resolution volume near detections |
| **Long-tail collapse** | 33.3 % of ScanNet200 classes never recalled | Small, rare objects; 0.08 m voxels still coarse relative to a "power strip" or "headphones" | Not a fusion problem — a resolution + class-imbalance problem |
| **Objects flush with other structure** | stove 0.236 (flush with counter), dishwasher AR .88 / AP .20 (flush with cabinetry) | High recall, low precision ⇒ found but massively over-detected. Geometric gating gives no help when the object *is* the surface | Semantic gating in addition to geometric gating |
| **Ceiling on geometry shaping** | GT-depth upper bound 58.8/33.4 vs ImGeoNet 54.8/28.4 **[PAPER]** Table 6 | `S` is predicted from features, not measured | Better surface estimation; multi-view photometric consistency; the residual **4.0/5.0** gap is the target |
| **Volatile late-training** | ScanNet ep 9→10→11→12: 54.52, 54.69, 54.46, 54.57 (±0.12) **[LOG]** | Batch size 1 (ScanNet), `RepeatDataset ×3` | **[INF] Report ±0.2 as run noise; differences below that are meaningless** |
| **ScanNet200 unconverged** | log stops at ep 11/30 **[LOG]** | Training abandoned | Rerun to 30 epochs — **an easy, honest contribution if you get GPU access** |

---

## 17. Limitations

**Methodological** — uniform mean over views, no per-view weighting; variance never reaches the detector; `S` is view-independent so it cannot model per-view occlusion; nearest-neighbour back-projection quantises geometry; fixed 6.4 m volume; axis-aligned boxes on ScanNet; **depth required at training time** (so it is not depth-free, only depth-free *at inference*).

**Dataset** — indoor only; ScanNet boxes derived from semantic labels, not annotated; ARKit depth is low-quality (the paper says so); ScanNet200's tail is too sparse for meaningful AP on ~1/3 of classes.

**Computational** — 485 MB model; ~490 ms/scene at 50 views; 10.8–19.4 GB training memory; ScanNet200 needed >1 day/run on 3×3090 and was not finished.

**Generalisation** — no cross-dataset experiment (train ScanNet → test ARKit); no unseen-viewpoint-distribution test; camera poses assumed accurate.

**Implementation** — pinned to torch 1.7.1 + mmcv 1.2.7 + mmdet 2.10.0, all long-deprecated and effectively Linux-only; custom CUDA op needs `nvcc`; **no pretrained checkpoints released**; ScanNet log came from a different code revision than the release (`ImVoxelNet` vs `ImGeoNet` class name).

---

## 18. Research gaps — specific

**Gap 1 — Views are averaged uniformly, despite the paper's own evidence that views differ enormously in value.**
**[PAPER]** Table 4 shows 10 views give 39.0 mAP and 100 give 62.4 — so views *do* carry very different amounts of information — yet Eq. 4 weights all of them `1/valid_count`. Nothing in the model estimates which of the 50 sampled frames actually sees an object well.
> **Direction:** predict a **per-(voxel, view) reliability weight** `w_t` — from feature magnitude, projected footprint size, or a small confidence head — and replace the mean with a weighted mean. Permutation-invariance and N-agnosticism are preserved. The 20-train/50-test protocol makes this cheap to validate.

**Gap 2 — Cross-view disagreement is computed and then thrown away.**
**[CODE]** `V_var` feeds `occ_head` only; the detector sees `avg_vols * occ`. Variance is the *only* signal in the whole model that knows whether views agree.
> **Direction:** concatenate `V_var` (or a richer set-statistic vector: mean, var, max, `valid_count`) into `neck_3d`'s input. A **one-line, ~1-day experiment** with a shipped matched baseline to compare against. **[INF] The single cheapest experiment in your whole project with a plausible payoff.**

**Gap 3 — Marginal value of a view is never modelled, only measured.**
The 10→100 curve saturates hard (+14.1 mAP for views 10→20; +2.2 for 50→100). The paper reports this but the *model* has no notion of redundancy: 100 near-duplicate frames cost 100 back-projections and buy almost nothing.
> **Direction:** an **adaptive view-selection policy** — score candidate frames for complementarity (e.g. pose diversity + predicted-coverage gain) and stop when marginal information falls below a threshold. Target: **match 50-view accuracy with ~25 selected views**, i.e. ~2× inference speed-up at equal mAP. The Table 4 curve gives you the baseline to beat for free.

**Gap 4 — The oracle-depth ceiling is 4.0 / 5.0 mAP above the method and unexplained.**
**[PAPER]** Table 6 (g-4) vs (g-1). The paper notes "there is room for improving Geometry Shaping Network in the future" but does not diagnose *why*.
> **Direction:** diagnose it. Is the gap in the *occupancy accuracy* (`acc_occ ≈ 0.93` **[LOG]** — what do the 7 % errors look like?) or in the *multiplicative gating form*? Try residual gating `V·(1+S)`, or supervising a signed distance rather than binary occupancy.

**Gap 5 — Thin structures fail catastrophically and the cause is architectural, not incidental.**
**[LOG]** picture AP@.25 = 0.043; tv_monitor 0.036; showercurtain loses 95.8 % of its AP going from IoU 0.25 to 0.5.
> **Direction:** sub-voxel geometry — predict occupancy *and* a within-voxel offset/normal, or use trilinear back-projection instead of `.round()`. A clean, well-motivated, well-scoped contribution with an obvious metric to move.

---

## 19. What to borrow

| Useful idea | Why it works | Limitation | How to adapt |
|---|---|---|---|
| **Mean over views normalised by `valid_count`** | Permutation-invariant, N-agnostic, and correctly handles partial visibility. **This is why train-20 / test-50 works at all** | Uniform weighting | **Adopt as your default fusion**, then add learned weights on top. It is strictly better engineered than MVDet's concat |
| **Train with few views, test with many** | Cheap training, strong test-time accuracy; also a built-in generalisation test | Only valid for symmetric pooling | **Copy this protocol.** It is free evidence that your fusion generalises across N |
| **Cross-view VARIANCE as a geometry cue** | Feature agreement across views is the classical MVS surface signal, obtained here for **+26 ms constant** instead of a depth network's 15.7× | Only used for the occupancy head | **Take it further** — feed variance to the recogniser too (Gap 2) |
| **Multiplicative geometric gating `V_g = S ⊙ V`** | Soft free-space suppression; no threshold; reuses the same features | `S` is view-independent | Consider a **per-view** gate `S_t` for occlusion-aware fusion |
| **Depth as *training-only* supervision** | Gives geometry priors without a depth sensor at deployment | Still needs RGB-D data to train | Excellent pattern: **supervise with a modality you will not have at test time** |
| **Shipping a matched-config baseline** (`imvoxelnet_*.py`) | Makes the ablation exactly reproducible; only the studied component differs | — | **Copy this discipline.** Your ablations should be config diffs, not code branches |
| **Reporting a views-vs-accuracy curve (Table 4)** | Turns "more views help" into a quantified, saturating curve | Uniform sampling only | **Reproduce this figure for your model**, and add a *selected*-views curve above it |

---

## 20. Verdict

**ImGeoNet is my recommendation as your primary baseline** for the multi-viewpoint branch, for four reasons that no other paper in your set matches:

1. **It is the only one whose fusion is genuinely N-agnostic and permutation-invariant** — so you can run Experiments A–D without touching the architecture.
2. **It already publishes the views-vs-accuracy curve** your Experiment A needs, so you have a free baseline to beat.
3. **It ships matched baseline configs and real training logs**, so paper claims can be checked (I did — §12, and they hold to within 0.2–0.6 points).
4. **Its weakness is precisely your topic:** uniform view averaging with no informativeness estimation, and a discarded disagreement signal.

**Practical cost:** it is the hardest environment to stand up (torch 1.7.1 / mmcv 1.2.7 / CUDA 11.0 / Linux / `nvcc`), and **no checkpoints are released**, so you must train from scratch — ~6 h on 4×P40 for ScanNet, ~13 h on one 3090 for ARKit. **[INF] Start with ARKitScenes**: single-GPU, `samples_per_gpu=8`, 12 epochs, and the log proves it completes on one RTX 3090 at 19.4 GB.

**Cross-references:** P1's concat is what its mean pooling improves on · P2's max-pool is a weaker statistic than its mean+variance · P5's cross-attention is the learned weighting it lacks · P6 uses the same lift-and-pool family in the outdoor/BEV setting.
