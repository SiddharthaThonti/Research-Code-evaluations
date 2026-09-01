# P5 — Cross-view Transformers for Real-time Map-view Semantic Segmentation

**Analysis date:** 2026-08-29
**Evidence base:** paper PDF (CVPR 2022, 10 pp.) + full source (`cross_view_transformers/`)
**Execution status:** ❌ not executed (§11). No number in this report was produced by running this code.

Tags: **[PAPER]** claimed · **[CODE]** verified in source · **[OBS]** measured by me · **[INF]** my inference

---

## 1. Inventory

| Item | Value |
|---|---|
| Title | Cross-view Transformers for real-time Map-view Semantic Segmentation |
| Authors | Brady Zhou, Philipp Krähenbühl (UT Austin) |
| Venue / Year | **CVPR 2022** (arXiv 2205.02833v1) |
| Code | `cross_view_transformers/` — clean, small, PyTorch-Lightning + Hydra. **~1 200 lines of model code total** |
| README | Yes (4.8 KB), with dataset-setup and label-generation docs |
| Datasets | nuScenes (60 GB) + authors' generated map-view labels (361 MB `.tar.gz`) — **neither present** |
| Pretrained weights | **None released** (no link in README, no `.pth` on disk) |
| Configs | ✅ **12 Hydra YAMLs** — `config.yaml`, `model/cvt.yaml`, `data/nuscenes*.yaml`, `experiment/*`, `loss/*`, `metrics/*` |
| Train / eval | `scripts/train.py`, `scripts/benchmark.py`, `scripts/overfit.py`, `scripts/generate_data.py`, `scripts/view_data.py`, 2 notebooks |
| Requirements | `requirements.txt` — torch 1.11.0, pytorch-lightning 1.6.2, hydra-core 1.1.1, efficientnet-pytorch 0.7.1, fvcore, einops, wandb |
| Splits | ✅ shipped: `data/splits/nuscenes/{train,val,mini_train,mini_val}.txt` |
| Result files | None (no logs, no checkpoints) |
| Reproducibility | **Blocked** (no GPU, no nuScenes, no weights) — but **by far the easiest of the five repos to stand up** |

---

## 2. Relevance to *Multi-View Object Recognition*

**A. Problem solved.** Predict a binary **map-view (BEV) semantic segmentation** — vehicles and/or drivable area — from 6 surround-view cameras on a car, in real time.

**B. What "multi-view" means here.** **Six fixed, calibrated cameras** in a 360° surround rig on one ego-vehicle. Views are *simultaneous* and *non-overlapping-ish* (marginal overlap). *N* = 6 nominally, but the architecture is **N-agnostic** (§7). Not multi-viewpoint object capture; not active view selection.

**C. Input.** `(I_k, K_k, R_k, t_k)ᵏ₌₁ⁿ`, `I_k ∈ ℝ^{H×W×3}`, resized to **224 × 480** with `top_crop: 46` **[CODE]** `config/data/nuscenes.yaml`. Intrinsics and extrinsics are *inverted* and consumed as positional embeddings, not as a warp.

**D. Output.** `y ∈ {0,1}^{h×w×C}` — a **200 × 200** binary map covering **100 m × 100 m** centred on the ego-vehicle. Classes are *vehicle* and *drivable area* (binary masks). **⚠ No object instances, no boxes, no categories.**

**E. Relevance score.**

| Reading | Score | Why |
|---|---|---|
| *Multi-view **fusion mechanism** for recognition* | **5 / 5** | ⭐ **CVT's cross-view attention is the single most transferable fusion mechanism in your entire set.** It is N-agnostic, permutation-invariant, geometry-aware, and performs **implicit learned view weighting** — exactly the capability every other paper lacks |
| *Multi-view **object recognition** as a task* | **2 / 5** | Output is a binary semantic mask, not object identity or category. No instances |

**Net: 4 / 5 — recommended as your *architectural* reference**, not your task reference. **[INF] Borrow the mechanism; take the task from P3.**

---

## 3. Research problem

- **Input:** n monocular views with calibration.
- **Output:** binary BEV mask `y ∈ {0,1}^{h×w×C}` in the orthographic map-view frame.
- **Assumptions:** synchronised cameras; known intrinsics/extrinsics; **calibration is roughly static across the dataset** (the paper says so — "camera calibration stays mostly static").
- **Constraints:** real-time (≥ 30 FPS on one RTX 2080 Ti).
- **Objective:** maximise IoU against orthographically projected 3D box annotations.

---

## 4. Motivation and the exact gap **[PAPER]** §1–2

Prior BEV methods build an **explicit geometric bridge** between camera and map view:
- **Depth-based** (Lift-Splat-Shoot, pseudo-LiDAR): estimate depth, unproject. *"Image-based depth estimates scale poorly with the distance to the observer"* and depth error dominates far-field accuracy.
- **Homography / flat-ground** (OFT, PON, VPN, monolayout): assume a plane. *"a fairly inflexible and rigid bottleneck"*.
- **Learned-implicit without geometry** (VED, VPN): no camera model at all → *"they forgo the inherent inductive biases contained in a calibrated camera setup"*.

**The gap:** either you hard-code geometry (rigid, error-prone) or you ignore it (throws away a free prior). **CVT's answer: encode calibration as a *positional embedding* and let attention learn the mapping.** *"We show that implicit geometric reasoning performs as well as explicit geometric models."*

---

## 5. Previous approaches **[PAPER]** §2

| Category | Examples | What it does | Limitation named |
|---|---|---|---|
| Monocular 3D detection | CenterNet, ROI-10D, pseudo-LiDAR | Detect in image, lift with depth | Depends entirely on monocular depth quality |
| Depth estimation | Eigen, monodepth, DORN, MiDaS | Regress depth | Camera-dependent, needs fusion of noisy estimates |
| Homography / plane | Abbas & Zisserman, Kim & Kum, Sengupta | Flat-ground warp | Breaks on non-planar scenes |
| Encoder–decoder implicit | VED, **VPN** (view relation module) | MLP maps camera features to BEV | *"do not model the geometric structure... forgo the inductive biases of a calibrated setup"* |
| **Explicit geometric reasoning** | **OFT**, **Lift-Splat-Shoot**, PON, **FIERY**, STA | Average-pool along a pillar (OFT), or soft-depth weighted splat (LSS) | LSS's *"attention weights are derived from geometric principles and not learned from data"* |

**[INF] The pivotal framing:** the paper explicitly positions LSS's depth-weighted splat as *an attention mechanism whose weights are not learned*. CVT's contribution is to **learn those weights instead**. That framing is the reason this paper matters for you.

---

## 6. Proposed method

```
 I₁..I_n (224×480)
    │
 EfficientNet-B4 (shared) → 2 scales:  reduction_2 (28×60, 8× down)
                                       reduction_4 (14×30, 16× down)
    │
    │  learned BEV positional embedding  c⁽⁰⁾ ∈ ℝ^{128×25×25}
    │  (25×25 = 200/2³, since decoder has 3 upsample blocks)
    ▼
 ┌────────────────── CrossViewAttention  ×2 (one per scale) ───────────────────┐
 │  camera embedding   τ_k   = cam_embed( E_k⁻¹[..., -1] )     ← camera CENTRE │
 │  image direction    δ_k,i = img_embed( E_k⁻¹ K_k⁻¹ x_i )                    │
 │  KEY   = normalize(δ_k,i − τ_k)  +  feature_proj(φ_k,i)                     │
 │  VALUE = feature_linear(φ_k,i)                                              │
 │  QUERY = normalize(bev_embed(world_xy) − τ_k)  +  c                          │
 │                                                                             │
 │  dot  = ⟨q, k⟩ / √d          per camera                                      │
 │  dot  = rearrange 'b n Q K -> b Q (n K)'    ← FLATTEN ACROSS CAMERAS         │
 │  att  = softmax(dot, dim=-1)                ← ★ SOFTMAX OVER ALL VIEWS ★     │
 │  z    = att @ v  (+ skip)  → LN → MLP → LN                                  │
 └──────────────────────────────┬──────────────────────────────────────────────┘
                                │  2 ResNet bottlenecks after each CVA block
                                ▼
                    Decoder: 3 × (bilinear ×2 + conv), residual
                                ▼
                     200 × 200 logits  (vehicle / road)
```

### Module detail **[CODE]** `model/encoder.py`

| Module | Lines | Operation |
|---|---|---|
| `EfficientNetExtractor` | `backbones/efficientnet.py` | EfficientNet-B4, taps `reduction_2` + `reduction_4` |
| `BEVEmbedding` | 66–111 | learned `nn.Parameter(σ·randn(128, 25, 25))`; also stores the ego-frame world grid via `V⁻¹` |
| `cam_embed` | 218, 244–246 | `Conv2d(4,128,1)` applied to `E_k⁻¹[..., -1:]` — the **camera centre in world coords** |
| `img_embed` | 217, 248–253 | `Conv2d(4,128,1)` applied to `d = E_k⁻¹ (K_k⁻¹ · pixel)` — the **ray direction** |
| key construction | 255–256, 267 | `img_embed = normalize(d_embed − c_embed)`; `key = img_embed + feature_proj(φ)` |
| query construction | 258–262, 274 | `bev_embed = normalize(w_embed − c_embed)`; `query = bev_embed + x` |
| **`CrossAttention.forward`** | **132–176** | see §7 |
| `Decoder` | `decoder.py` | 3 × (`interpolate ×2` + conv), `residual: True`, blocks `[128,128,64]` |

**[INF] The elegance:** Eq. 2's geometric cosine similarity
`sim_k(x^I, x^W) = ⟨R_k⁻¹K_k⁻¹x^I , x^W − t_k⟩ / ‖·‖‖·‖`
is *literally* the attention dot-product once both sides are replaced by MLP embeddings (Eq. 3). Geometry is not approximated — it is **rewritten as the attention operation itself**. Depth never appears, because a ray direction plus an unknown depth is exactly what a normalised direction vector encodes.

---

## 7. ⭐ Multi-view fusion strategy — the most important section in your whole analysis

**Mechanism: SOFTMAX CROSS-ATTENTION jointly over (views × image patches).**

**[CODE]** `encoder.py:156-161` — the three lines that matter:

```python
dot = self.scale * torch.einsum('b n Q d, b n K d -> b n Q K', q, k)   # per-camera scores
dot = rearrange(dot, 'b n Q K -> b Q (n K)')                            # FLATTEN cameras
att = dot.softmax(dim=-1)                                               # ONE softmax over ALL
a   = torch.einsum('b Q K, b K d -> b Q d', att, v)                     # v is already (b, n·h·w, d)
```

Each BEV cell `Q` attends over the **union** of all patches from all cameras, normalised by a **single softmax**.

### Why this is the right design — [INF], five properties, all verifiable above

| Property | Why it follows | Which other papers have it |
|---|---|---|
| **Permutation-invariant** | Softmax over a flattened set; camera order changes only the ordering of terms in a sum | P2 (max), P3 (mean), P6 (sum) — but they achieve it by *discarding* information |
| **N-agnostic** | No weight depends on `n`; `n` only lengthens the key/value set | P2, P3, P6 |
| **Learned, content-dependent view weighting** | The softmax *competes* views against each other per BEV cell. A view that cannot see a cell receives low logits and is automatically suppressed | ⭐ **Only CVT.** P1/P4 concat uniformly; P2 max; P3 mean; P6 sum |
| **Geometry-aware without warping** | Query and key both carry calibration-derived embeddings; alignment is *learned*, not imposed | P4 tries calibration-free but drops geometry entirely |
| **Handles partial visibility natively** | **[PAPER]** §3.2: *"not every map-view location has a corresponding image patch in each view. Front-facing cameras do not see the back... We allow the attention mechanism to select both camera and location within each camera"* | — |

**[PAPER]'s own ablations quantify how much each ingredient matters (Tables 3 & 4, Setting 2 IoU):**

| Ablation | IoU | Δ from full |
|---|---|---|
| **Full model** | **36.0** | — |
| No camera-aware embedding δ (image features only as keys) | 31.0 | **−5.0** ← largest |
| No image features φ in the keys (pure geometry) | 33.2 | −2.8 |
| No map-view embedding refinement (single fixed embedding) | 33.6 | −2.4 |
| No positional embedding at all | 31.0 | −5.0 |
| Learned per-camera embedding (not calibration-derived) | 34.4 | −1.6 |
| Camera-aware + Random Fourier projection | 35.8 | −0.2 |
| Camera-aware + Linear projection (default) | **36.0** | — |

**[INF] Reading these together:** neither geometry alone (33.2) nor appearance alone (31.0) suffices; the product is worth +5.0 over appearance-only. And a *learned per-camera* embedding gets 34.4 — only 1.6 below the calibration-derived one. **That is a striking result for your research:** it means **~89 % of the calibration benefit can be recovered from a learned per-camera code**, which is a semi-calibration-free path that P4 (CaMuViD) never considers.

### Weaknesses — [INF]

1. **O(n · h · w · H · W) attention.** Cost is linear in *n* but quadratic in resolution. The paper mitigates with `w = h = 25` map-view queries — **[PAPER]** §4: *"For computational efficiency, we choose w = h = 25 as the cross-attention function grows quadratically with grid size."* Coarse BEV queries are a real limitation (GeoBEV/P6 attacks exactly this).
2. **Softmax normalises away absolute evidence.** Attention weights sum to 1 whether *one* view weakly sees a cell or *six* views strongly agree. **[INF] Like max-pooling in P2, this discards the "how many views corroborate" signal — though unlike max, at least the *relative* weighting is learned.**
3. **Only 2 scales, 2 attention blocks.** Not a deep transformer.
4. **Single-timestep.** No temporal fusion (deliberate: they compare to "FIERY static").
5. **Calibration assumed near-static.** The learned-embedding ablation (34.4) suggests the model partly memorises the rig.

---

## 8. Loss function **[CODE]** `losses.py`, `config/loss/*.yaml`

```
L = Σ_k  w_k · L_k        (MultipleLoss, weights from YAML)
```
- `BinarySegmentationLoss` = **sigmoid focal loss** (`fvcore.nn.sigmoid_focal_loss`), `alpha = −1.0` (i.e. **disabled**), `gamma = 2.0`.
- `CenterLoss` = same focal loss on a **center heatmap** (used in `cvt_nuscenes_vehicle`, which overrides `loss: center_loss`).
- `min_visibility` masking: **[CODE]** `losses.py:52-54` — `loss = loss[mask[:, None]]` where `mask = batch['visibility'] >= min_visibility`. **Labels below a nuScenes visibility level are excluded from the loss entirely.**

**[INF] Two notes.** (i) `alpha = -1.0` means no class balancing — only the focal `gamma` modulation. (ii) The visibility mask is important and easy to miss: **the same masking is applied in `metrics.py:65-71`**, so the *reported IoU is computed only over sufficiently-visible cells*. Any comparison to another method must use the same `min_visibility`, or the numbers are not comparable.

---

## 9. Dataset

| | nuScenes |
|---|---|
| Scenes / frames | 1000 scenes × 40 keyframes = **40 k samples** |
| Cameras | **6** (`CAM_FRONT_LEFT, CAM_FRONT, CAM_FRONT_RIGHT, CAM_BACK_LEFT, CAM_BACK, CAM_BACK_RIGHT`) **[CODE]** `nuscenes_dataset.py:111-112` |
| Coverage | full 360° |
| Input res | 224 × 480, `top_crop = 46` |
| BEV label | 200 × 200 over 100 m × 100 m |
| Labels | orthographic projection of 3D boxes → binary mask; 12 classes generated, grouped by `label_indices` |
| Splits | shipped: `train.txt` (8.2 KB), `val.txt` (1.8 KB), plus `mini_*` and 2 qualitative lists |
| Augmentation | `augment: 'none'` by default **[CODE]** `data/nuscenes.yaml:27` |
| Argoverse 1.1 | mentioned in paper Table 2; README says labels *"coming soon™"* — **never released** |

**Two evaluation settings** **[PAPER]** §5:
- **Setting 1** — 100 m × 50 m @ **25 cm** (Roddick et al.)
- **Setting 2** — 100 m × 100 m @ **50 cm** (Philion & Fidler / LSS) — **used for all ablations**

**[INF] Representative of your problem?** Only structurally. It gives you a genuine multi-camera rig with real calibration and a huge, diverse dataset — but binary masks, not object recognition, and a 60 GB download plus a 361 MB label archive.

---

## 10. Experimental setup **[PAPER]** §4 + **[CODE]** `config/config.yaml`

| | Paper | Code | Match |
|---|---|---|---|
| Backbone | EfficientNet-B4, pretrained, fine-tuned | `model_name: efficientnet-b4`, `layer_names: [reduction_2, reduction_4]` | ✔ |
| Scales R | 2 — (28,60) and (14,30) | `middle: [2, 2]` (2 bottlenecks per scale) | ✔ |
| BEV embedding | `w = h = 25`, `D = 128` | `dim: 128`; 25 = 200/2³ from `decoder.blocks` length 3 | ✔ |
| Attention | 4 heads, `dim_head = 64` | ⚠ **`heads: 4, dim_head: 32`** in `model/cvt.yaml` | ✖ **mismatch** |
| Decoder | 3 × (bilinear up + conv) → 200×200 | `blocks: [128,128,64]`, `factor: 2`, `residual: True` | ✔ |
| Optimiser | AdamW, one-cycle | AdamW `lr 4e-3`, `wd 1e-7`; OneCycleLR `div_factor 10, pct_start 0.3` | ✔ |
| Batch | 4 per GPU | `batch_size: 4` | ✔ |
| Epochs / steps | **30 epochs** | **`max_steps: 30001`** | ⚠ different units |
| Precision | not stated | `precision: 32` | — |
| Grad clip | not stated | `gradient_clip_val: 5.0` | — |
| Seed | not stated | `seed: 2022` | — |
| Hardware | 4 GPUs, ~8 h (≈ 32 GPU-hours) | `gpus: -1` (all available) | ✔ |
| Inference | **35 FPS**, 1 × RTX 2080 Ti | `scripts/benchmark.py` provided | ✔ |
| Params | **5 M** | — | — |

**[INF] Two documented discrepancies:**
1. **`dim_head`: paper says 64, config says 32.** With 4 heads that is 128 vs 256 inner dimension — a real difference. `dim = 128`, so `dim_head = 32` keeps `heads·dim_head = dim = 128` (the standard convention), which suggests **the config is right and the paper's "64" is an error**, or the released config differs from the paper run.
2. **"30 epochs" vs `max_steps: 30001`.** At 4×4 = 16 samples/step and ~28 k train samples, 30 001 steps ≈ **17 epochs**, not 30. The README's *"An average job of 50k training iterations takes ~8 hours"* mentions yet a third figure (50 k). **[INF] The released config does not reproduce the paper's stated schedule.** If you reproduce CVT, expect to tune `max_steps`.

---

## 11. Attempt to execute — failed, documented

```
OS      Windows 11        Python 3.14.3 (conda base only)
PyTorch NOT INSTALLED     GPU: NONE (nvidia-smi not found)
```

| Problem | Cause | Solution attempted | Result |
|---|---|---|---|
| No PyTorch / no CUDA GPU | Machine has neither | `import torch`; `nvidia-smi` | ❌ |
| No nuScenes | 60 GB keyframes + map expansion; must be downloaded from nuscenes.org (registration required) | Searched tree for `nuscenes`/`samples` dirs | ❌ Absent |
| No map-view labels | `cvt_labels_nuscenes.tar.gz` (361 MB) from UT Austin | Same | ❌ Absent |
| **No pretrained weights** | **None ever released** — README has no checkpoint link | Searched tree for `*.pth`/`*.ckpt` | ❌ Zero |
| Argoverse labels | README: *"coming soon™"* | — | ❌ Never released; Table 2's Argoverse row is unreproducible |
| Old pinned stack | torch 1.11.0 + pytorch-lightning 1.6.2 + hydra 1.1.1 | Read `requirements.txt` | ⚠ PL 1.6 API differs sharply from modern PL; `compute_on_step` in `metrics.py:12` is **removed in torchmetrics ≥ 0.11** |

**Nothing was run.**

**[INF] Silver lining — this is the most reproducible repo of the five.** Pure `pip`, **no custom CUDA ops**, no MATLAB, no mmcv/mmdet pinning, single GPU sufficient, and `scripts/overfit.py` exists for a fast sanity check. If you get *any* GPU, **start here.** The binding constraint is the 60 GB dataset, not the software.

---

## 12. Results reported (paper claims, not reproduced)

**Table 1 — nuScenes vehicle map-view segmentation (IoU)**

| Method | Setting 1 | Setting 2 | Params (M) | FPS |
|---|---|---|---|---|
| PON | 24.7 | – | 38 | 30 |
| VPN | 25.5 | – | 18 | – |
| STA | 36.0 | – | – | – |
| Lift-Splat | – | 32.1 | 14 | 25 |
| FIERY (static) | **37.7** | 35.8 | 7 | 8 |
| **CVT** | 37.5 | **36.0** | **5** | **35** |

**Table 2 — vehicle + drivable area**

| | Vehicle | Drivable |
|---|---|---|
| OFT | 30.1 | 71.7 |
| Lift-Splat | 32.1 | 72.9 |
| **CVT (nuScenes)** | **36.0** | **74.3** |
| Monolayout (Argoverse) | 32.1 | 58.3 |
| PON (Argoverse) | 31.4 | 65.4 |
| **CVT (Argoverse)** | **35.2** | **73.6** |

**[INF] The honest reading of Table 1:** CVT is **not** the most accurate — FIERY wins Setting 1 (37.7 vs 37.5). CVT's claim is the **Pareto frontier**: equal accuracy at **1/1.4 the parameters** (5 M vs 7 M), **4.4× the FPS** (35 vs 8), and **1/3 the training cost** (32 vs 96 GPU-hours). The paper states this plainly. That is a well-supported and unusually honest claim.

| Metric | Paper reported | My reproduction | Difference | Explanation |
|---|---|---|---|---|
| Setting 1 IoU | 37.5 | **not attempted** | — | No GPU, no nuScenes (60 GB), no weights (§11) |
| Setting 2 IoU | 36.0 | not attempted | — | " |
| FPS | 35 | not attempted | — | " |

---

## 13. ⭐ Controlled experiments — CVT has the best *config-level* support of any repo here

**[CODE]** `config/data/nuscenes.yaml:10` — `cameras: [[0, 1, 2, 3, 4, 5]]`, consumed by `NuScenesDataset.parse_scene(scene_record, camera_rigs)` (line 132–134), which loops `for cam_idx in camera_rig` (line 163).

**This is a list of *camera rigs*.** Changing it changes which cameras form a sample — **with zero code modification**, because the architecture is N-agnostic.

| Experiment | Command | Why it works |
|---|---|---|
| **A. Number of views** | `data.cameras=[[1]]`, `[[1,4]]`, `[[0,1,2]]`, `[[0,1,2,3,4,5]]` | Attention set-size is free |
| **B. View selection** | `data.cameras=[[1]]` (front) vs `[[4]]` (back) vs `[[0,2]]` (front-diagonals) | Directly names cameras |
| **C. View order** | `data.cameras=[[5,3,1,0,4,2]]` | **Should be bit-identical** — softmax over a flattened set is permutation-invariant. **A perfect positive control for your invariance claims** |
| **D. Missing views** | Train on 6, evaluate on subsets | **[PAPER]** Fig. 4 already does this (see below) |
| **E. View quality** | `data/augmentations.py` exists | Insert corruptions |
| **Data augmentation** | `augment: 'none'` → other | Config flag |
| **Multi-rig training** | `cameras: [[0,1,2],[3,4,5]]` | Generates *two* samples per frame — a built-in **train-time view-count augmentation**. **[INF] Almost certainly unexplored, and directly on your topic** |

### **[PAPER] Fig. 4 — camera-dropout robustness (the result you need)**

Take a model trained on all 6 cameras; randomly drop *m* ∈ {0,1,2,3} at eval:

| Cameras dropped | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| IoU (read from Fig. 4) | ~36 | ~31 | ~27.5 | ~22.5 |

**[PAPER]** §5.5: *"the performance decreases linearly with the number of cameras dropped. This is quite intuitive as different cameras only overlap marginally. Thus each removed camera reduces the visible area linearly... the transformer-based model is generally quite robust to this camera dropout and the overall performance does not degrade beyond unobserved parts of the scene."*

**[INF] Contrast this with P4/CaMuViD** — the sharpest cross-paper comparison available to you:

| | CVT (nuScenes, 6 surround cams) | CaMuViD (Wildtrack, 7 overlapping cams) |
|---|---|---|
| Camera overlap | **marginal** | **heavy** — **[OBS]** mean 5.26 of 7 cameras see each person; C1↔C6 Jaccard 92.8 % |
| Effect of dropping views | **linear** degradation | **steeply non-linear**: 7→6 cams *improves* MODA (95.0 → 95.6); 3→1 cams collapses it (90.6 → 60.1) |
| Why | Each camera owns a disjoint angular sector; losing one loses exactly that sector | Views are largely redundant for *coverage*, so extra views contribute *disambiguation* with diminishing and eventually negative returns |

**[INF] The generalisable law:** *the shape of the accuracy-vs-view-count curve is determined by the geometric redundancy of the rig, not by the fusion method.* Neither paper states this; it emerges only from reading them together. **This is a genuinely novel observation you can build on** — and it implies that any claim like "our fusion scales better with views" is meaningless unless the rig's redundancy is reported.

---

## 14. Performance / cost

| | CVT | FIERY (static) | Lift-Splat |
|---|---|---|---|
| Params | **5 M** | 7 M | 14 M |
| FPS (RTX 2080 Ti) | **35** | 8 | 25 |
| Training | **32 GPU-h** | 96 GPU-h | — |
| Setting 2 IoU | **36.0** | 35.8 | 32.1 |

**Answer to "accurate because of a good multi-view strategy, or because it's big?"**
**[INF] Emphatically the strategy — CVT is the strongest case in your whole set.** It is the **smallest** model in Table 1 and the **fastest**, while matching or beating everything. **[PAPER]** §5.1 makes the control explicit: *"We intentionally use the same image feature extractor (EfficientNet-B4) and similar decoder architecture as FIERY. This suggests our cross-view transformer is capable of combining features from multiple views in a more efficient manner."* Same backbone, same decoder, different fusion → equal accuracy at 4.4× the speed. That is a clean, controlled attribution of the gain to the fusion mechanism.

---

## 15. Ablations available

| Ablation | Available? | How |
|---|---|---|
| No camera-aware embedding | ✅ | `bev_embed`/`cam_embed` are separable modules |
| No image features in keys | ✅ **config flag** | `cross_view.no_image_features: True` **[CODE]** `encoder.py:208-214` |
| No map-view refinement | ✅ | `skip: False` + reduce `middle` |
| Random Fourier vs Linear projection | ⚠ | `RandomCos` class exists (`encoder.py:49-63`) but is **not wired into any config** — the Table-4 Fourier row needs a small edit |
| Number of scales | ✅ | `backbone.layer_names` + `middle` |
| Heads / dim_head | ✅ | `cross_view.heads`, `cross_view.dim_head` |
| Visibility threshold | ✅ | `metrics: visibility_metrics` vs `default_metrics` |
| Loss (BCE vs center) | ✅ | `loss: default_loss` vs `center_loss` |

---

## 16. Failure analysis

| Failure type | Evidence | Probable cause | Potential solution |
|---|---|---|---|
| **Far-field degradation** | **[PAPER]** Fig. 3: IoU falls from ~36 at 0 m to **0 by ~60 m**; CVT is *below* FIERY beyond ~50 m | **[PAPER]** §5.4: *"Partially occluded far-away samples have fewer corresponding image features, thus learning a mapping from map-view to camera-view directly is harder: There is less training data and fewer geometric priors to rely upon"* | Distance-aware sampling/loss; explicit depth prior in the far field; more data |
| Occluded / distant vehicles missed | **[PAPER]** §5.6: *"accurately segments nearby vehicles, but does not sense far away or occluded vehicles well"* | Attention needs *some* image evidence; fully occluded cells have none | Temporal fusion; amodal supervision |
| Coarse 25×25 BEV queries | **[PAPER]** §4: attention *"grows quadratically with grid size"*; final 200×200 comes from 3 upsample blocks | Compute-bound design choice | Deformable/linear attention; GeoBEV (P6) attacks exactly this |
| Linear degradation under dropout | **[PAPER]** Fig. 4 | Marginal camera overlap — **[INF] a property of the rig, not the method** | Correct behaviour; no fix needed |
| Softmax discards evidence count | **[CODE]** `att = dot.softmax(-1)`, weights sum to 1 | Normalisation removes absolute magnitude | Add an un-normalised gate or a `valid_count`-style channel |
| No instances / no categories | Binary mask output | Task design | Add a detection head (this is what your work would need) |

---

## 17. Limitations

**Methodological** — binary semantic mask, no instances/categories; single timestep; 25×25 map-view queries limit spatial precision; attention cost quadratic in grid size; softmax normalises away corroboration count; calibration assumed near-static.
**Dataset** — nuScenes only in practice (Argoverse labels never released); `augment: 'none'` by default; 6-camera surround rig is one specific geometry.
**Computational** — modest by design (5 M params, 35 FPS) — **the least concerning of your six**.
**Generalisation** — no cross-dataset experiment; the learned-per-camera ablation (34.4 vs 36.0) hints the model partly memorises the rig.
**Implementation** — **no pretrained weights released**; requires 60 GB nuScenes; pinned to PL 1.6.2 / torchmetrics 0.7.2 (the `compute_on_step` kwarg in `metrics.py:12` is removed in torchmetrics ≥ 0.11 → will crash on modern installs); `dim_head` paper/config mismatch; `max_steps: 30001` ≠ the paper's "30 epochs"; `RandomCos` implemented but unused.

---

## 18. Research gaps

**Gap 1 — Cross-view attention learns *which* view to trust but never reports *how much* it is trusted.**
The attention weights `att` (**[CODE]** `encoder.py:158`) are a per-(BEV-cell, view, patch) informativeness measure — computed on every forward pass and **immediately discarded**. The paper visualises them qualitatively (Fig. 6) but never *uses* or *analyses* them.
> **Direction:** aggregate `att` over patches to get a **per-(cell, view) attention mass**, and study it: does the model learn the rig's geometry? Does it down-weight occluded views? Can the attention mass be *supervised* (with visibility labels, which nuScenes provides and CVT already loads for `min_visibility`) to make view selection explicit and controllable? **[INF] This is a low-cost, high-value, genuinely novel study — the data is already in the forward pass.**

**Gap 2 — Softmax normalisation destroys the corroboration signal.**
Weights sum to 1 regardless of whether one view weakly sees a cell or six views strongly agree.
> **Direction:** add an **un-normalised confidence channel** — e.g. `logsumexp(dot)` before softmax, or a learned gate — so the decoder knows *how much total evidence* supports each cell, not just its distribution. Directly analogous to ImGeoNet's (P3) `valid_count` normalisation and its variance channel.

**Gap 3 — Calibration is 89 % replaceable by a learned code, and nobody has followed up.**
**[PAPER]** Table 4: learned-per-camera embedding **34.4** vs calibration-derived **36.0**.
> **Direction:** a **semi-calibration-free** model — learn per-camera embeddings, optionally initialised from rough calibration, and fine-tune them on a new rig with few labels. This is the principled version of what P4 (CaMuViD) attempts by removing geometry entirely. Success criterion: transfer to a new rig with ≤ 1.6 IoU loss and no calibration.

**Gap 4 — Multi-rig training is supported by the config and never used.**
**[CODE]** `cameras: [[0,1,2],[3,4,5]]` would emit two 3-camera samples per frame.
> **Direction:** train with **randomised camera subsets** as augmentation and test whether it improves dropout robustness and N-generalisation. One config line; nobody has published it.

**Gap 5 — The dropout curve's shape is a property of the rig, never controlled for.**
> **Direction (cross-paper, §13):** define and report a **rig redundancy statistic** (e.g. mean views per object, pairwise view Jaccard — both of which I computed for Wildtrack/MultiviewX in §13) alongside every view-count curve. This makes "our method scales better with views" a falsifiable claim for the first time.

---

## 19. What to borrow

| Useful idea | Why it works | Limitation | How to adapt |
|---|---|---|---|
| ⭐ **Softmax attention over the *union* of (views × patches)** | Simultaneously permutation-invariant, N-agnostic, and a **learned per-cell view selector** — the only mechanism in your six papers with all three properties | Quadratic in grid size; normalises away evidence count | **Make this your fusion backbone.** Then fix the count problem (Gap 2) |
| ⭐ **Camera calibration as a positional embedding, not a warp** | Keeps the geometric prior while staying differentiable and tolerant to calibration error. Worth **+5.0 IoU** over appearance-only keys | Assumes near-static calibration | Generalises to *any* multi-view setting — including object-centric capture, where the "camera pose" is the object viewpoint |
| **Direction-vector embedding `normalize(d − c)`** | Encodes a ray without committing to a depth — sidesteps monocular depth error entirely | Depth remains implicit; far-field suffers | **[INF] The cleanest answer in your set to "how do I fuse views without depth?"** |
| **Query = `bev_embed − cam_embed`** (per-camera query) | Lets one BEV cell ask a *different question* of each camera, conditioned on where that camera is | — | Directly reusable |
| **Iterative refinement across scales** | Coarse attention localises, fine attention sharpens. Worth **+2.4 IoU** | Only 2 scales | Combine with P2's multi-scale idea → multi-scale *and* attention-weighted |
| **`no_image_features` as a config flag** | Ablations are config diffs, not code branches | — | **Copy this practice** (as ImGeoNet does with matched configs) |
| **Reporting params + FPS + GPU-hours next to accuracy** | Makes the Pareto claim falsifiable and turns a 2nd-place IoU into a 1st-place contribution | — | **Copy this reporting protocol.** It is how you defend a method that is not SOTA on raw accuracy |
| **`cameras` as a config-level rig specification** | Makes view-count/selection/order experiments free | Unused by the authors | **Exploit it** (Gap 4) |

---

## 20. Verdict

**CVT is your architectural blueprint.**

- **Adopt** its cross-view attention as the fusion mechanism. It is the only design in your six papers that is simultaneously permutation-invariant, N-agnostic, geometry-aware, and capable of **learned per-cell view weighting** — the exact capability whose absence defines the gaps in P1, P2, P3, P4 and P6.
- **Do not** adopt its task: binary BEV segmentation is not object recognition. Pair CVT's encoder with P3's (ImGeoNet's) recognition setting.
- **Start here if you get a GPU.** No custom CUDA ops, no mmcv pinning, single-GPU trainable, `overfit.py` for sanity checks. The only real barrier is the 60 GB nuScenes download — and note **no checkpoints were released**, so you must train (~32 GPU-hours for the paper's schedule).
- **Fix before use:** `torchmetrics` `compute_on_step` (removed ≥ 0.11), and reconcile `max_steps: 30001` with the paper's "30 epochs".

**Cross-references:** P1/P4's uniform concat is what its attention replaces · P2/P3/P6's max/mean/sum are the un-learned special cases of its softmax · **[INF] the combination your research should target is CVT's attention (learned weighting) + ImGeoNet's set statistics (evidence count) + MSMVD's multi-scale projection — no published method has all three.**
