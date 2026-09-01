# Master Comparison of the Six Papers

Tags: **[PAPER]** claimed · **[CODE]** verified in source · **[LOG]** from shipped training logs · **[OBS]** measured by me · **[INF]** inference

---

## Table 1 — Overview

| | **P1 MVDet** | **P2 MSMVD** | **P3 ImGeoNet** | **P4 CaMuViD** | **P5 CVT** | **P6 GeoBEV** |
|---|---|---|---|---|---|---|
| Venue / Year | ECCV 2020 | arXiv 2025 (BMVC fmt) | ICCV 2023 | CVPR 2025 | CVPR 2022 | AAAI 2025 |
| Task | MV pedestrian detection | MV pedestrian detection | **MV 3D object detection** | MV pedestrian detection | Map-view segmentation | **MV 3D object detection** |
| Dataset | Wildtrack, MultiviewX | GMVD, WT, MVX | ScanNetV2, ScanNet200, ARKit | Wildtrack, MultiviewX | nuScenes (+Argoverse) | nuScenes |
| Input | N RGB + calib | N RGB + calib | T RGB + K,T pose | N RGB, **no calib** | 6 RGB + calib | 6 RGB + calib (+T frames) |
| Output | BEV occupancy (1 class) | BEV occupancy ×3 scales | **3D box + 18/189/17 classes** | per-view 2D boxes | binary BEV mask | **3D box + 10 classes** |
| # views | **7 / 6 (fixed)** | 6–7, **varies (GMVD)** | **20 train / 50 test, swept 10–100** | **7 / 6 (fixed)** | 6 (N-agnostic) | 6 (+2 or 8 frames) |
| Backbone | ResNet-18 (dilated) | ResNet-18 + PAFPN | ResNet-50 + FPN | InternImage-T (DCNv3) | EfficientNet-B4 | ResNet-50/101/VoV-99 |
| **Cross-view fusion** | **channel CONCAT** | **MAX per scale** | **MEAN + variance** | **channel CONCAT** | ⭐ **softmax ATTENTION** | **SUM** |
| Attention | ✗ | ✗ | ✗ | ✗ (only within-view) | ⭐ **✓ cross-view** | ✗ |
| Learned view weighting | ✗ | ✗ (max = hard select) | ✗ | ✗ | ⭐ **✓ implicit** | ✗ |
| Validity / visibility norm. | ✗ (zero-pad) | ✗ | ⭐ **✓ `/valid_count`** | ✗ | ✓ implicit (softmax) | ✗ |
| Permutation-invariant | ✗ | ✓ | ✓ | ✗ | ✓ | ✓ |
| N-agnostic architecture | ✗ | ✓ | ⭐ **✓ (train 20 / test 50)** | ✗ | ✓ | ✓ |
| Multi-scale image feats | ✗ | ⭐ **✓ (3 levels)** | ✓ FPN (fused before lift) | ✓ FPN (per level, independent) | ✓ 2 scales | ✓ (`scale_num=2`, within view) |
| Headline result | 88.2 MODA (WT) | 80.2 MODA (GMVD) | 54.8 mAP@.25 (ScanNet) | 95.0 MODA (WT)† | 36.0 IoU (nuSc) | 0.579 mAP / 0.662 NDS |
| Params | ~30 M (19 M in fusion) | 22.9 M | 485 MB (~127 M) | >60 M | ⭐ **5 M** | not reported |
| Speed | not reported | ❌ **not reported** | 490 ms/scene @50 views | not reported | ⭐ **35 FPS** | 17–25 FPS |
| **Code present** | ✅ | ❌ **none** | ✅ | ✅ | ✅ | ✅ |
| **Weights present** | ✗ (linked) | ✗ | ✗ **(none exist)** | ✗ (linked) | ✗ **(none exist)** | ✗ (linked) |
| **Logs / results present** | ✗ | ✗ | ⭐ **✅ 3 full logs** | ✅ GT annotations | ✗ | ✗ |
| Relevance to your topic | **3.5 / 5** | **3 / 5** (1/5 practically) | ⭐ **4 / 5** | **2.5 / 5** | ⭐ **4 / 5** (5/5 for mechanism) | **2.5 / 5** |

† not protocol-comparable to the rest of its own table — see §5.

---

## Table 2 — Architectural ideas

| Paper | Feature extractor | View representation | **Fusion strategy** | View weighting | View selection | Loss | Key innovation |
|---|---|---|---|---|---|---|---|
| **P1 MVDet** | ResNet-18, dilated, 1 scale | Perspective-warped feature map on `z=0` ground plane | `torch.cat` over N → `(512N+2)` ch → 3 dilated convs | none | none | Gaussian-MSE (ground) + α·per-view head/foot MSE | Anchor-free **feature** perspective transform; CNN replaces CRF |
| **P2 MSMVD** | ResNet-18 + PAFPN, 3 scales | 3 BEV features, one per image scale, each from 5 height slices | **`max` over views, per scale**, then BEV-PAFPN across scales | none (max = winner-take-all) | none | Focal (3 occupancy maps) + L1 (3 offset maps) | **Multi-Scale Projection** — BEV features inherit their source scale |
| **P3 ImGeoNet** | ResNet-50 + FPN | Back-projected voxel volume (40³-ish @0.16 m) | ⭐ **masked `mean` (÷ valid_count)** + **variance** side-channel; then `V_g = S ⊙ V` | none (uniform mean) | uniform `linspace` sampling | Focal(cls) + CE(centerness) + 3D-IoU(box) + **10×Focal(occupancy)** | **Geometry shaping** — predict surface probability from the same features; depth supervises training only |
| **P4 CaMuViD** | InternImage-T (DCNv3) + FPN | ⚠ **channel-mixed feature map, no spatial warp** | `torch.cat` over N → `Conv1×1(256N→256)` → per-view back-projection | none | none | Cascade R-CNN losses + **1e-4** · cycle-consistency `L_vp` | **Calibration-free** learned projection / back-projection matrices |
| **P5 CVT** | EfficientNet-B4, 2 scales | Camera-aware positional embeddings (ray direction − camera centre) | ⭐ **softmax cross-attention over (views × patches) jointly** | ⭐ **learned, per BEV cell** | implicit (attention) | Sigmoid focal (BEV) + focal (center), visibility-masked | **Calibration as positional embedding** — geometry learned, never warped |
| **P6 GeoBEV** | ResNet-50/101/VoV-99 + CustomFPN | Radial (polar) BEV `ℝ^{C×D×W}` → bilinear → Cartesian | **`.sum(1)` over 6 cameras**; `cat` over T frames | within-view only (`depth_weight ⊙ fg_weight`) | none | CenterHead + BCEFocal depth + BCEFocal fg + **CAI loss** | **In-Box Label** (supervise object *interior*) + **RC-Sampling** (gather, not scatter) |

---

## Table 3 — Reproducibility status (Step 1 inventory, verified this session)

| ID | Paper | Year | Main task | Dataset | Code | Pretrained | Data on disk | Logs | **Reproducibility status** |
|---|---|---|---|---|---|---|---|---|---|
| P1 | MVDet | 2020 | MV pedestrian detection | Wildtrack, MultiviewX | ✅ | ✗ (OneDrive link) | ✗ | ✗ | ❌ **Blocked** — needs **2 GPUs** (`cuda:0`+`cuda:1` hard-coded), MATLAB for official metric, both datasets |
| P2 | MSMVD | 2025 | MV pedestrian detection | GMVD, WT, MVX | ❌ **none** | ✗ | ✗ | ✗ | ❌ **Impossible** — no implementation exists |
| P3 | ImGeoNet | 2023 | MV 3D object detection | ScanNetV2/200, ARKit | ✅ | ✗ **(never released)** | ✗ (splits only) | ⭐ **✅ 3 logs** | ❌ **Blocked** — torch 1.7.1+cu110, mmcv 1.2.7, `nvcc`, Linux, TB-scale data |
| P4 | CaMuViD | 2025 | MV pedestrian detection | Wildtrack, MultiviewX | ✅ | ✗ (HF link) | ⭐ **✅ COCO GT** | ✗ | ❌ **Blocked** — DCNv3 CUDA op, mmcv 1.5.0, no images, 2-stage training |
| P5 | CVT | 2022 | Map-view segmentation | nuScenes | ✅ | ✗ **(never released)** | ✗ (splits only) | ✗ | ❌ **Blocked** — 60 GB nuScenes + 361 MB labels. ⭐ **Easiest software stack: pure pip, no CUDA ops, 1 GPU** |
| P6 | GeoBEV | 2025 | MV 3D object detection | nuScenes | ✅ | ✗ (Drive link) | ✗ | ✗ | ❌ **Blocked** — `bev_pool_v2` CUDA op, nuScenes + lidarseg + nuImages masks, **8-GPU** recipe |

**Environment verified this session:** Windows 11 · Python 3.14.3 (conda `base` only) · **PyTorch not installed** (`ModuleNotFoundError: No module named 'torch'`) · **no NVIDIA GPU** (`nvidia-smi` not found) · no MATLAB.
**Total `.pth` / `.ckpt` files found across all six folders: 0.**

---

## Table 4 — What each paper reports about *number of views*

| Paper | View-count experiment | Result | Comment |
|---|---|---|---|
| **P1 MVDet** | ✗ none | — | Architecture fixed to N; cannot vary without retraining |
| **P2 MSMVD** | ✗ none | — | Trains on GMVD where N varies **across scenes**, but never reports N-vs-accuracy |
| **P3 ImGeoNet** | ⭐ **✅ 10→100 views** (Table 4, ARKit) | 39.0 → 53.1 → 57.1 → 59.5 → 60.2 → 61.8 → **62.4** mAP@.25 | **Saturating.** +14.1 for 10→20; **+2.2 for 50→100**. Geometry shaping ≈ worth 2.5× the views (40 views ≈ ImVoxelNet @100) |
| **P4 CaMuViD** | ✅ **camera elimination 1→7** (Table 3) | 60.1 → 77.8 → 90.6 → 93.8 → 93.8 → **95.6** → 95.0 MODA | ⚠ **Non-monotonic — 6 cameras beat 7.** [CODE] Implemented by feeding **black images**, not retraining |
| **P5 CVT** | ✅ **camera dropout 0→3** (Fig. 4) | ~36 → ~31 → ~27.5 → ~22.5 IoU | **Linear** degradation; *"does not degrade beyond unobserved parts of the scene"* |
| **P6 GeoBEV** | ✗ none | — | No camera-count or dropout study at all |

### ⭐ [INF] The cross-paper law that emerges (stated by none of them)

> **The shape of the accuracy-vs-view-count curve is determined by the geometric redundancy of the camera rig, not by the fusion method.**

| Rig | Redundancy | Curve shape |
|---|---|---|
| nuScenes surround (P5, P6) | **Low** — cameras tile 360° with marginal overlap | **Linear** decay under dropout: each lost camera = one lost sector |
| Wildtrack / MultiviewX (P1, P2, P4) | **Very high** — **[OBS]** mean **5.26 of 7** cameras see each pedestrian; **C1↔C6 Jaccard 92.8 %** | **Steep then flat, then negative**: 1→3 cameras is decisive; 4–5 add nothing to coverage; the 7th **hurts** |
| ScanNet / ARKit (P3) | High but *sequential* — a moving camera revisits | Smooth **saturating** curve |

**Consequence:** any claim of the form *"our fusion scales better with views"* is untestable unless rig redundancy is reported. **Recommendation: report mean-views-per-object and pairwise view-Jaccard alongside every view-count curve.** I computed both for Wildtrack/MultiviewX in `04_results/E1_*`.

---

## Table 5 — ⚠ Evaluation protocols are **not** interchangeable

| Paper | Metric | TP criterion | Comparable to? |
|---|---|---|---|
| P1 MVDet | MODA/MODP | BEV point within **0.5 m** (MOTChallenge MATLAB devkit) | P2 ✓ |
| P2 MSMVD | MODA/MODP | BEV point within **0.5 m** | P1 ✓ |
| **P4 CaMuViD** | MODA/MODP | ⚠ **2D image-space box IoU ≥ 0.45** + Hungarian, **per identity across all views** **[CODE]** `evaluation.py:43-116` | ⚠ **Not P1/P2** |
| P3 ImGeoNet | mAP@0.25/0.50 | 3D IoU | own line |
| P5 CVT | IoU @ {0.4, 0.5} | pixel-wise, **visibility-masked** | own line |
| P6 GeoBEV | mAP / NDS | nuScenes centre-distance | own line |

**[INF]** P4's Table 1 places its image-space numbers (95.0 MODA) directly beside BEV-protocol numbers from MVDet/MVDeTr/MVFP. The paper *does* disclose the difference (§4.2) but frames it around MODP only, while MODA/precision/recall are equally affected. **If you use CaMuViD as a baseline, re-evaluate it under the standard BEV protocol.**

---

## Table 6 — Cost vs. accuracy honesty

| Paper | Reports params? | FLOPs? | Latency? | Memory? | Parameter-matched control? |
|---|---|---|---|---|---|
| P1 MVDet | ✗ | ✗ | ✗ | ✗ | ⭐ **✓ (same net, 3 projection choices: 26.8 / 68.2 / 88.2)** |
| P2 MSMVD | ✓ | ✗ | ❌ | ❌ | ⭐ **✓ (backbone sweep R18→R101, Table 4)** |
| P3 ImGeoNet | ✓ (485 MB) | ✗ | ⭐ **✓ (Table 5, per view count)** | ✓ **[LOG]** 10.8–19.4 GB | ✓ (matched `imvoxelnet_*.py` configs shipped) |
| P4 CaMuViD | ✗ | ✗ (`get_flops.py` exists) | ✗ (timing `print`s in code) | ✗ | ✗ |
| P5 CVT | ⭐ **✓ 5 M** | ✗ | ⭐ **✓ 35 FPS** | ✗ | ⭐ **✓ same backbone+decoder as FIERY** |
| P6 GeoBEV | ✗ | `get_flops.py` shipped | ⭐ **✓ FPS per BEV size** | ✗ | ⭐ **✓ 0-param modules; plug-in on 3 baselines** |

**[INF] Best-practice models to copy:** **P5** (params + FPS + GPU-hours beside accuracy, same backbone as the competitor) and **P6** (zero-parameter contributions validated as plug-ins across three independent baselines).

---

## Table 7 — Where each paper's contribution actually sits

| Axis of the multi-view pipeline | P1 | P2 | P3 | P4 | P5 | P6 |
|---|---|---|---|---|---|---|
| Per-view feature extraction | | ⭐ multi-scale | | ⭐ DCNv3 | | ⭐ multi-scale depth |
| **Image → 3D/BEV lift** | ⭐ feature warp | ⭐ multi-scale + height slices | ⭐ geometry shaping | ⚠ *no spatial lift* | ⭐ implicit via attention | ⭐ RC-Sampling |
| **Cross-view fusion** | concat | max | mean+var | concat | ⭐ **attention** | sum |
| Spatial reasoning after fusion | ⭐ large-kernel dilated | ⭐ BEV-FPN | 3D convs | FRM (per view) | ResNet bottlenecks | BEV ResNet |
| Supervision design | aux per-view | multi-scale deep sup. | ⭐ depth→occupancy | cycle-consistency | visibility masking | ⭐ **In-Box + CAI** |
| Recognition head | 1-class heatmap | 1-class heatmap | ⭐ **18/189 classes** | Cascade R-CNN ×N | binary mask | ⭐ **10 classes** |

**[INF] Read the middle row.** Of six papers spanning 2020–2025 on "multi-view" recognition, **exactly one (P5) contributes anything to cross-view fusion itself.** The other five innovate on *lifting* and *supervision*, and reuse concat / max / mean / sum unchanged. **That is your gap, and it is a wide one.**
