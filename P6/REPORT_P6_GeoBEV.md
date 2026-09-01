# P6 — GeoBEV: Learning Geometric BEV Representation for Multi-view 3D Object Detection

**Analysis date:** 2026-08-29
**Evidence base:** paper PDF (AAAI 2025, pp. 9960–9968) + full source (`GeoBEV/`, a BEVDet/mmdetection3d fork)
**Execution status:** ❌ not executed (§11). No number in this report was produced by running this code.

Tags: **[PAPER]** claimed · **[CODE]** verified in source · **[OBS]** measured by me · **[INF]** my inference

---

## 1. Inventory

| Item | Value |
|---|---|
| Title | GeoBEV: Learning Geometric BEV Representation for Multi-view 3D Object Detection |
| Authors | Jinqing Zhang¹, Yanan Zhang¹, Yunlong Qi², Zehua Fu³, Qingjie Liu¹˒³˒⁴*, Yunhong Wang¹˒³ (¹Beihang SKLVRTS ²Beijing Jingwei Hirain ³Hangzhou Innovation Inst. ⁴Zhongguancun Lab) |
| Venue / Year | **AAAI 2025** (arXiv 2409.01816) |
| Code | `GeoBEV/` — full BEVDet-style mmdetection3d fork (~250 Python files) |
| README | Yes (4.4 KB) with a results table and Google-Drive model links |
| Datasets | nuScenes + nuScenes-lidarseg + nuImages-predicted instance masks — **none present** |
| Pretrained weights | Google Drive links (models + nuImages backbones + `samples_instance_mask`) — **not on disk** |
| Configs | ✅ `configs/geobev/` (3 official) + **`configs/my/` (11 further variants incl. distillation experiments)** |
| Train / eval | `tools/dist_train.sh`, `tools/dist_test.sh`, `tools/create_data_bevdet.py`, `tools/generate_point_label.py` |
| Analysis tools | ✅ `tools/analysis_tools/{get_flops,benchmark,benchmark_view_transformer,analyze_logs,vis}.py` |
| Requirements | README: torch 1.10.0+cu111, mmcv-full 1.5.3, mmdet 2.27.0, mmsegmentation 0.25.0, `pip install -e .` |
| **Custom CUDA op** | ⚠ `mmdet3d/ops/bev_pool_v2/src/{bev_pool.cpp, bev_pool_cuda.cu}` — needs `nvcc` |
| Result files | None (no logs, no checkpoints, no `work_dirs/`) |
| Reproducibility | **Blocked** (no GPU, no nuScenes, no weights, custom CUDA op) |

---

## 2. Relevance to *Multi-View Object Recognition*

**A. Problem solved.** 3D object detection (10 nuScenes classes) from 6 surround-view cameras, via a Bird's-Eye-View representation — targeting the **geometric quality** of that BEV representation.

**B. What "multi-view" means here.** **Six fixed, calibrated surround cameras** on a car, plus **temporal** views (2 or 8 adjacent frames). Marginal inter-camera overlap. *N* = 6, fixed by the dataset.

**C. Input.** 6 camera images at **256 × 704** (R50) or 512 × 1408 (R101), with intrinsics/extrinsics; `Ncams` and `multi_adj_frame_id_cfg = (1, 1+1, 1)` for the 2-frame model, `(1, 1+7, 1)` for the 8-frame "longterm" model **[CODE]**.
**Training also requires:** LiDAR point clouds (for depth labels) and **nuImages-predicted instance masks** (for foreground gating) — both dropped at inference.

**D. Output.** 3D boxes + **10 object categories** + velocity + attributes, via a CenterPoint head. Metrics: mAP and NDS.

**E. Relevance score.**

| Reading | Score | Why |
|---|---|---|
| *Multi-camera 3D object recognition* | **3 / 5** | Correct task shape (multi-view in → categories + 3D boxes out), strong engineering |
| *Multi-view **fusion** research* | **2 / 5** | ⚠ **GeoBEV's contribution is not about view fusion at all.** Its cross-view fusion is a plain `.sum(1)` inherited unchanged from LSS/BEVDet. The novelty is in *per-view lifting quality* and *depth supervision* |

**Net: 2.5 / 5.** **[INF] This is the paper most easily mis-filed as "multi-view fusion research".** Read it instead as: *how to make the image→BEV lift geometrically accurate*. That is a genuinely useful, but different, question.

---

## 3. Research problem

- **Input:** 6 surround images + calibration (+ T temporal frames).
- **Output:** 3D boxes with categories.
- **Assumptions:** known calibration; ego-motion for temporal alignment; **LiDAR + instance masks available at training**.
- **Constraints:** BEV resolution vs compute — the central tension of the paper.
- **Objective:** maximise NDS/mAP on nuScenes.

---

## 4. Motivation and the exact gap **[PAPER]**

> *"the geometric quality of BEV representation has never received sufficient attention, and the limitation of low-resolution representation always arises."*

Two named problems (paper Fig. 1):

1. **Feature vacancy in LSS-style methods.** LSS/BEVDet/BEVDepth pool *pseudo-points* into BEV cells. Cells that receive no pseudo-point stay **empty**, and *"the sparsity will further increase along with the BEV resolution"* — so you cannot simply raise resolution (Fig. 1b shows it getting *worse*).
2. **Transformer-based BEV is quadratically expensive**, so its resolution is also capped.
3. **Voxel-Sampling** (strict projection then height-squeeze, e.g. SimpleBEV, TPVFormer) needs huge intermediate 4D frustum tensors.

Plus a **supervision** problem: LiDAR depth labels record only the **surface facing the ego car**, not the object's actual 3D extent. *"The lack of objects' complete geometric information hinders the subsequent BEV encoder and detection head from precisely recognizing their size and orientation."*

**Three contributions:** **RC-Sampling** (efficient dense high-res BEV), **In-Box Label** (supervise the object's *interior*, not just its front surface), **CAI Loss** (weight interior points by proximity to the centroid).

---

## 5. Previous approaches **[PAPER]**

| Category | Examples | What it does | Limitation named |
|---|---|---|---|
| Uniform-depth lift | OFT | All voxels on a ray share features | Ignores depth distribution |
| **Depth-distribution lift (LSS family)** | **LSS**, BEVDet, BEVDet4D, **BEVDepth**, BEVStereo, TiG-BEV, SA-BEV, BEV-IO, FB-BEV, BEVNeXt | Predict per-pixel depth distribution, splat pseudo-points, pool into BEV | **Feature vacancy**, worsening with resolution; depends on custom pooling kernels |
| **Transformer BEV** | BEVFormer(V2), PolarFormer, DFA3D | Deformable cross-attention retrieves image features per BEV query | Cost escalates rapidly with BEV resolution |
| **Voxel-Sampling** | SimpleBEV, TPVFormer | Sample along strict projection into voxels, squeeze height | Many sampling ops + huge intermediate features |
| Sparse-query (no explicit BEV) | DETR3D, PETR(V2), StreamPETR, Sparse4D(V2), RayDN | Object queries attend to image features directly | *"omission of explicit BEV representation causes geometric information loss, limiting their precision upper bound"* |
| LiDAR-supervised depth | BEVDepth, CaDDN, BEVStereo | Project LiDAR to supervise depth | Labels only the ego-facing **surface** |

---

## 6. Proposed method

```
 6 surround images (256×704)
        │
   ResNet-50 (nuImages-pretrained) + CustomFPN   → I ∈ ℝ^{C×H×W}
        │                                        → D ∈ ℝ^{D×H×W}  (DepthNet, D=118 bins over [1,60] m @0.5 m)
        │                                        → fg ∈ ℝ^{1×H×W} (foreground score)
        │
   ┌────┴──── RC-Sampling ────────────────────────────────────────────┐
   │  context_weight = depth_weight ⊙ fg_weight                        │
   │      depth_weight = where(softmax(D) ≥ 1/D, sigmoid(D), 0)        │
   │      fg_weight    = (sigmoid(fg) ≥ 0.1)                           │
   │                                                                   │
   │  ★ Radial BEV:   Bᴿ = [(I → ℝ^{W×C×H}) ⊗ (D → ℝ^{W×H×D})] → ℝ^{C×D×W}
   │      i.e. a batched matmul that CONTRACTS the height dim H  (Eq. 3)│
   │                                                                   │
   │  ★ Cartesian:    Bᶜ(x,y) = BilinearSample(Bᴿ, Project(x,y))  (Eq. 4)
   │      F.grid_sample from the polar (D,W) grid to the (x,y) grid     │
   └────┬──────────────────────────────────────────────────────────────┘
        │
   bev_feat = grid_sample(...).view(B, N, C, h, w).sum(1)      ← ★ CROSS-VIEW FUSION = SUM
        │
   temporal: bev_feat = torch.cat(bev_feat_list, dim=1)        ← CONCAT over T frames
        │
   CustomResNet BEV backbone + FPN_LSS neck
        │
   CenterHead → 3D boxes + 10 classes
```

### Module detail **[CODE]**

| Module | File : line | Operation |
|---|---|---|
| `DepthNet` | `necks/rcsample.py:41-112` | predicts `D` depth bins **+ 1 foreground channel**, conditioned on an MLP of the calibration (`mlp_input`) |
| multi-scale depth | `rcsample.py:349-358` | `scale_num=2`: refine depth at `downsamples=[16, 8]` via `interpolate ×2` + conv |
| gating | `rcsample.py:360-365` | `depth_weight = where(softmax(D) ≥ 1/D, sigmoid(D), 0)`; `fg_weight = sigmoid(fg) ≥ 0.1` (`keep_threshold`) |
| **Radial BEV (Eq. 3)** | `rcsample.py:366-367` | `torch.matmul(context_weight.permute(0,3,1,2), context.permute(0,3,2,1))` → contracts `H` |
| **Cartesian resample (Eq. 4)** | `rcsample.py:377` | `F.grid_sample(frustum_feat.view(B*N,C,D,W), norm_bev_coor)` |
| **cross-view fusion** | **`rcsample.py:378`** | **`bev_feat.view(B, N, out_ch, h, w).sum(1)`** |
| temporal fusion | `detectors/geobev.py:142` | `torch.cat(bev_feat_list, dim=1)` |
| In-Box Label + CAI loss | `detectors/geobev.py:165-230` | builds `gt_bboxes_inbox`, returns `loss_fg`, `loss_inbox` |

**[INF] Why RC-Sampling is elegant:** LSS *scatters* pseudo-points into BEV cells (some cells get none → vacancy). RC-Sampling instead builds a dense **radial/polar** feature map `Bᴿ ∈ ℝ^{C×D×W}` — dense *by construction*, since every (depth-bin, column) pair is filled by the matmul — and then **gathers** into Cartesian BEV by bilinear sampling. Gathering cannot leave holes. **[PAPER]**: *"Bilinearly sampling `Bᴿ` instead of pooling the sparse pseudo-points guarantees that each position in `Bᶜ` has valid features."*

---

## 7. Multi-view fusion strategy — and why it is *not* this paper's contribution

**Mechanism: SUM over the 6 views.** **[CODE]** `rcsample.py:378`

```python
bev_feat = F.grid_sample(frustum_feat.view(B * N, C, D, W), norm_bev_coor)
bev_feat = bev_feat.view(B, N, self.out_channels, h, w).sum(1)     # ← the entire cross-view fusion
```

**Properties — [INF]:**

| Property | Status |
|---|---|
| Permutation-invariant | ✅ (sum) |
| N-agnostic | ✅ architecturally |
| Learned view weighting | ❌ **none** |
| Validity/visibility normalisation | ❌ **none** — unlike ImGeoNet (P3), there is no division by a valid-view count |
| Evidence-count preserved | ⚠ implicitly (sum grows with overlap) but not disentangled from magnitude |

**[INF] This is inherited unchanged from LSS/BEVDet.** The paper never discusses cross-view aggregation, never ablates it, and never mentions the word "fusion" in the multi-camera sense. **GeoBEV improves what each view contributes, not how views are combined.**

**Consequence — a real and unremarked weakness:** in a surround rig, adjacent cameras overlap slightly. In the overlap wedges the sum **doubles** the feature magnitude relative to singly-observed regions, purely as a function of geometry. **[INF] Contrast ImGeoNet (P3), which divides by `valid_count` for exactly this reason.** GeoBEV's BEV backbone must learn to compensate. With nuScenes' marginal overlap the effect is small; **for your research on heavily overlapping rigs (Wildtrack: [OBS] mean 5.26 of 7 cameras see each person) an unnormalised sum would be actively harmful.**

**Where the paper *does* do something view-adaptive:** `context_weight = depth_weight ⊙ fg_weight` is a **per-pixel, per-depth-bin** confidence applied *before* projection. It is genuinely a form of evidence weighting — but it is computed **within a single view**, with no knowledge of the other views.

---

## 8. Loss function **[PAPER]** Eqs. 5–7, **[CODE]** config

```
L = L_det(CenterHead)                                  # GaussianFocal (w 6.0) + L1 (w 1.5)
  + w_depth · L_depth(BCEFocal)                        # loss_depth_weight = [300, 600]
  + w_fg    · L_fg(BCEFocal)                           # loss_fg_weight    = [33, 67]
  + L_CAI                                              # the In-Box term
```

**Vanilla In-Box Label (Eq. 5):** a pseudo-point `p` is positive iff `p ∈ ⋃ᵢ Bᵢ` (inside *any* GT box) — so depth scores are encouraged to fill the object's **whole volume**, not just its front surface.

**Three corrections** **[PAPER]** Fig. 4 — this is the careful part:
- **(a) Occlusion between objects:** if object A occludes B, image pixels in the occluded region are *not supervised* (neither positive nor negative), letting the network decide.
- **(b) Background inside a box:** the **HTC model pretrained on nuImages** supplies instance masks; background pseudo-points inside a GT box are filtered out.
- **(c) Behind the background surface:** pseudo-points behind the LiDAR-observed surface are labelled **negative**, so the model learns "how thick the ground is".

**Centroid-Aware Inner (CAI) Loss (Eqs. 6–7):**
```
W_CAI = ∛[ min(f,b)/max(f,b) × min(l,r)/max(l,r) × min(u,d)/max(u,d) ]
L_CAI(p,y) = −(1−α)pᵞ log(1−p)                     if y = 0
             −W_CAI · α(1−p)ᵞ log(p)                if y = 1
```
`f,b,l,r,u,d` are distances to the six box faces; `W_CAI → 1` at the centroid, `→ 0` at faces. Sigmoid replaces Softmax so each depth bin is scored independently (binary, not multi-class).

**[PAPER] Table 5 — the loss ablation is unusually clean and additive:**

| Label | Sigmoid | Focal | CAI | mAP | NDS |
|---|:-:|:-:|:-:|---|---|
| LiDAR | | | | 0.337 | 0.456 |
| Vanilla In-Box | | | | 0.345 | 0.464 |
| Vanilla In-Box | ✓ | | | 0.347 | 0.466 |
| Vanilla In-Box | ✓ | ✓ | | 0.351 | 0.470 |
| **In-Box** (3 corrections) | ✓ | ✓ | | 0.356 | 0.474 |
| **In-Box** | ✓ | ✓ | ✓ | **0.359** | **0.478** |

**[INF]** Every step is +0.2 to +0.5 mAP; the whole depth-supervision redesign is worth **+2.2 mAP / +2.2 NDS** and, as the paper stresses, **adds zero parameters**. That is the most cost-effective contribution in the paper — and it is a *supervision* idea, transferable to any depth-lifting method.

---

## 9. Dataset

| | nuScenes |
|---|---|
| Scenes | 750 train / 150 val / 150 test |
| Cameras | **6** surround |
| Annotation | 2 Hz keyframes, 10 classes |
| Image size | 256×704 (R50) · 512×1408 (R101) · 640×1600 crop (VoVNet-99, test set) |
| BEV grid | x,y ∈ [−51.2, 51.2] @ **0.4 m** → 256×256; z ∈ [−5,3] @ 8 |
| Depth bins | [1.0, 60.0] @ 0.5 m → **118 bins** |
| Temporal | 2 frames (`(1,1+1,1)`) or **8 frames** ("longterm", `(1,1+7,1)`) |
| Sampling | **CBGS** (class-balanced grouping and sampling) |
| Augmentation | standard + **BEV-Paste** (`bev_paste=True`) |
| **Extra training data required** | ⚠ **nuScenes-lidarseg** (depth labels) **and** `samples_instance_mask` (HTC nuImages predictions) |
| Metrics | mAP, **NDS**, mATE/mASE/mAOE/mAVE/mAAE |

**[INF] Representative of your problem?** Only for the driving setting. Six cameras with marginal overlap, on-ground objects, and — critically — **three extra supervision modalities at training time** (LiDAR, lidarseg, instance masks). It is the heaviest data pipeline of your six papers by a wide margin.

---

## 10. Experimental setup **[PAPER]** + **[CODE]** `configs/geobev/geobev-r50-nuimage-cbgs.py`

| Item | Value |
|---|---|
| Backbone | ResNet-50 / ResNet-101 / VoVNet-99 (DD3D-pretrained, test set) |
| **Backbone init** | `load_from = 'ckpts/nuimage_pretrained_r50.pth'` — **nuImages-pretrained, not ImageNet** |
| Neck | `CustomFPN`, `out_channels=512` |
| View transform | `RCSample`, `scale_num=2`, `keep_threshold=0.1`, `downsamples=[16,8]`, `out_channels=80` |
| BEV encoder | `CustomResNet` (channels 160/320/640) + `FPN_LSS` → 256 |
| Head | `CenterHead` + `CenterPointBBoxCoder`, rotate-NMS |
| Optimiser | **AdamW, lr 2e-4, weight decay 0.075** |
| Schedule | `EpochBasedRunner, max_epochs=20`, step at [20] |
| `samples_per_gpu` | **8** |
| GPUs | **8** (`bash tools/dist_train.sh <cfg> 8`) → effective batch 64 |
| Loss weights | `loss_depth_weight=[300,600]`, `loss_fg_weight=[33,67]`, cls 6.0, bbox 1.5 |
| Ablation protocol | **[PAPER]**: R50, **24 epochs, no CBGS** |
| Reported hardware / runtime / memory | ❌ **not stated** (FPS is reported in Table 4) |

⚠ **Config/paper mismatch:** the released configs use `max_epochs=20` **with** CBGS, while the paper says main results are *"trained for 20 epochs with CBGS strategy"* ✔ but ablations use *"24 epochs without the CBGS strategy"* — **and no 24-epoch/no-CBGS config is shipped.** The ablation tables (3–5) are therefore not directly reproducible from the released configs.

---

## 11. Attempt to execute — failed, documented

```
OS      Windows 11        Python 3.14.3 (conda base only)
PyTorch NOT INSTALLED     GPU: NONE (nvidia-smi not found)
```

| Problem | Cause | Solution attempted | Result |
|---|---|---|---|
| No PyTorch / no CUDA GPU | Machine has neither | `import torch`; `nvidia-smi` | ❌ |
| **Custom CUDA op `bev_pool_v2`** | `mmdet3d/ops/bev_pool_v2/src/bev_pool_cuda.cu` compiled by `pip install -e .` | Confirmed source files present | ❌ Needs `nvcc` + CUDA toolkit |
| mmcv-full 1.5.3 / mmdet 2.27.0 | Pinned; Linux-first; no wheels for modern Python | Read README | ❌ |
| No nuScenes | Full trainval + test | Searched tree | ❌ |
| **No nuScenes-lidarseg** | Needed by `tools/generate_point_label.py` for depth+semantic labels | Read README §2d | ❌ |
| **No `samples_instance_mask`** | HTC-on-nuImages predictions; Google-Drive download | Read README §2b | ❌ Extra multi-GB dependency |
| No `.pkl` infos | Produced by `tools/create_data_bevdet.py` | Searched tree | ❌ |
| No backbone ckpt | `ckpts/nuimage_pretrained_r50.pth` | Searched tree for `*.pth` | ❌ Zero checkpoints anywhere |
| No model weights | Google Drive | Same | ❌ |
| Training needs **8 GPUs** | `dist_train.sh <cfg> 8`, `samples_per_gpu=8` | Read README | ❌ Far beyond a single-GPU budget |

**Nothing was run.** **[INF] GeoBEV has the heaviest setup cost of your six papers**: two custom CUDA ops' worth of build machinery, three data modalities, four downloads, and an 8-GPU training recipe.

---

## 12. Results reported (paper claims, not reproduced)

**Table 1 — nuScenes *val*** (★ = benefits from perspective-view pre-training)

| Method | Backbone | Image | Frames | mAP↑ | NDS↑ |
|---|---|---|---|---|---|
| BEVDet | R50 | 256×704 | 1 | 0.298 | 0.379 |
| BEVDepth | R50 | 256×704 | 2 | 0.351 | 0.475 |
| SA-BEV | R50 | 256×704 | 2 | 0.387 | 0.512 |
| **GeoBEV** | R50 | 256×704 | 2 | **0.415** | **0.535** |
| StreamPETR★ | R50 | 256×704 | 8 | 0.450 | 0.550 |
| RayDN★ | R50 | 256×704 | 8 | 0.469 | 0.563 |
| **GeoBEV★** | R50 | 256×704 | 8 | **0.479** | **0.575** |
| RayDN★ | R101 | 512×1408 | 8 | 0.518 | 0.604 |
| **GeoBEV★** | R101 | 512×1408 | 8 | **0.526** | **0.615** |

**Table 2 — nuScenes *test*:** GeoBEV (VoVNet-99, 640×1600, 8 frames) **0.579 mAP / 0.662 NDS**, vs StreamPETR 0.550/0.636 and RayDN 0.565/0.645.

**Table 3 — component ablation (R50, 24 ep, no CBGS)**

| Baseline | RC-Sampling | In-Box | mAP | NDS |
|---|:-:|:-:|---|---|
| BEVDepth | | | 0.337 | 0.456 |
| BEVDepth | ✓ | | 0.363 | 0.489 |
| BEVDepth | | ✓ | 0.359 | 0.478 |
| BEVDepth | ✓ | ✓ | **0.381** | **0.500** |
| BEVDet | | | 0.283 | 0.350 |
| BEVDet | ✓ | ✓ | 0.310 | 0.391 |
| BEVStereo | | | 0.354 | 0.474 |
| BEVStereo | ✓ | ✓ | **0.388** | **0.513** |

**Table 4 — RC-Sampling vs other transforms (the efficiency claim)**

| Method | BEV size | DS | mAP | NDS | **FPS** |
|---|---|---|---|---|---|
| BEVPoolv2 | 128×128 | 16 | 0.337 | 0.456 | 22.7 |
| BEVPoolv2 | 256×256 | 16 | 0.344 | 0.474 | 16.6 |
| DFA3D | 128×128 | 16 | 0.335 | 0.455 | 20.2 |
| DFA3D | 256×256 | 16 | 0.344 | 0.469 | 11.7 |
| Voxel-Sampling | 128×128 | 16 | 0.342 | 0.464 | 20.6 |
| Voxel-Sampling | 256×256 | 16 | 0.354 | 0.484 | 13.8 |
| **RC-Sampling** | 128×128 | 16 | 0.344 | 0.465 | **24.8** |
| **RC-Sampling** | 256×256 | 16 | 0.358 | 0.482 | **17.4** |
| **RC-Sampling** | 256×256 | **8** | **0.363** | **0.489** | 17.0 |

| Metric | Paper reported | My reproduction | Difference | Explanation |
|---|---|---|---|---|
| val mAP (R50, 2f) | 0.415 | **not attempted** | — | No GPU/PyTorch, no nuScenes, no lidarseg, no masks, no weights, `nvcc` needed, 8 GPUs (§11) |
| val NDS (R50, 2f) | 0.535 | not attempted | — | " |
| test NDS | 0.662 | not attempted | — | " |

**[INF] Assessment of the claims.**
- The **generality ablation (Table 3) is the strongest evidence**: the two modules improve **three different baselines** (BEVDepth +4.4 mAP, BEVDet +2.7, BEVStereo +3.4). Plug-in gains across independent architectures are much harder to fake than a single tuned number.
- **RC-Sampling's efficiency claim is well-supported** (Table 4): at 256×256 it is both the most accurate *and* the fastest (0.358 / 17.4 FPS vs Voxel-Sampling 0.354 / 13.8 and DFA3D 0.344 / 11.7).
- ⚠ **The `128 → 256` BEV gain is modest** (+1.4 mAP for RC-Sampling) at a 30 % FPS cost. The paper's motivating Fig. 1b ("larger BEV size makes LSS worse") is qualitative; Table 4 shows BEVPoolv2 *also* improves with resolution (0.337 → 0.344). **[INF] The vacancy problem is real but less catastrophic than Fig. 1 implies.**
- ⚠ Table 1 mixes rows with and without perspective-view pre-training (★). GeoBEV's headline R50 comparisons against StreamPETR/RayDN are ★-to-★, which is fair — but the unstarred GeoBEV row (0.415) should not be read against starred competitors.

---

## 13. Controlled experiments — support in this repo

| Experiment | Supported? | Notes |
|---|---|---|
| **A. Number of views** | ⚠ possible, undocumented | `data_config['Ncams']` exists and `.sum(1)` is N-agnostic, but the nuScenes pipeline assumes 6 named cameras. Not a config-level experiment like CVT's `cameras` list |
| **B. View selection** | ⚠ | Would need edits to `PrepareImageInputsGeoBEV` |
| **C. View order** | ✅ provably irrelevant | `.sum(1)` is permutation-invariant — a valid positive control |
| **D. Missing views** | ⚠ | No blanking/dropout mechanism provided |
| **E. View quality** | ✅ | Standard mmdet3d augmentation pipeline |
| **BEV resolution** | ✅ **first-class** | `grid_config['x'/'y']` step 0.4 → 0.2 etc. This is the paper's Table 4 |
| **Temporal frames** | ✅ **first-class** | `multi_adj_frame_id_cfg = (1, 1+k, 1)` — the *only* diff between the base and "longterm" configs (**verified by diffing the two files: 1 line**) |
| **View-transform swap** | ✅ | `RCSample` vs `LSSViewTransformerBEVDepth` etc. are registry entries |
| **Loss/label ablation** | ⚠ partial | `configs/my/` contains 11 extra variants (distillation experiments), but no 24-epoch/no-CBGS ablation config |
| **FLOPs / latency** | ✅ **shipped** | `tools/analysis_tools/get_flops.py`, `benchmark.py`, **`benchmark_view_transformer.py`** — the last one is purpose-built for Table 4 |

**[INF] GeoBEV is the best-instrumented repo of your six for *cost* measurement** (dedicated FLOPs and view-transformer benchmark scripts), and among the worst for *view-count* experiments.

---

## 14. Performance / cost

| Config | BEV | mAP | NDS | FPS |
|---|---|---|---|---|
| RC-Sampling 128² DS16 | 128×128 | 0.344 | 0.465 | **24.8** |
| RC-Sampling 256² DS16 | 256×256 | 0.358 | 0.482 | 17.4 |
| RC-Sampling 256² DS8 | 256×256 | **0.363** | **0.489** | 17.0 |

**[PAPER]** claims RC-Sampling *"reduces more than 90 % time cost and memory cost required by Voxel-Sampling while creating BEV representation with equal geometric quality"* — i.e. the 90 % figure refers to the **view-transform stage in isolation**, not end-to-end (end-to-end is 17.4 vs 13.8 FPS ≈ 26 % faster).

**Answer to "accurate because of a good multi-view strategy, or because it's big?"**
**[INF] Neither — it is accurate because of *better supervision and a better lift*, and the paper proves this well.**
- **In-Box Label + CAI Loss add exactly zero parameters** and are worth **+2.2 mAP / +2.2 NDS** (Table 5). That is the cleanest "not-just-capacity" result in your entire paper set.
- RC-Sampling is *faster* than what it replaces while being more accurate (Table 4).
- The gains transfer to three independent baselines (Table 3).
**However**, none of this is a *multi-view fusion* contribution (§7).

---

## 15. Failure analysis

No logs or predictions shipped; derived from the paper's reported metrics.

| Failure type | Evidence | Probable cause | Potential solution |
|---|---|---|---|
| **Orientation error is GeoBEV's weakest sub-metric** | **[PAPER]** Table 1: GeoBEV★ R101 mAOE **0.318** vs StreamPETR 0.315, RayDN 0.315 — GeoBEV wins mAP/NDS/mATE/mASE but **loses mAOE** | In-Box Label fills the box interior, improving *size* (mASE **0.254**, best) but orientation is under-constrained by an interior-filling objective | Orientation-aware inner weighting; the CAI weight is isotropic (∛ of three ratios) and carries no directional signal |
| **Velocity error** | mAVE 0.236 (R50 8f) — good but temporal-dependent | Needs 8 frames | Inherent to camera-only |
| Feature vacancy only partly solved | Table 4: 128→256 gives +1.4 mAP at −30 % FPS | Bilinear gather fills cells, but *information* is still limited by the depth-bin resolution (0.5 m) | Finer depth bins; learned adaptive binning |
| **Unnormalised cross-view sum** | **[CODE]** `rcsample.py:378` | Overlap wedges get doubled magnitude | Divide by a valid-view count (ImGeoNet/P3's fix) |
| Depends on predicted instance masks | **[PAPER]** *"We use the HTC pretrained on nuImages to provide the mask of objects"* | Mask errors propagate into In-Box supervision | Self-supervised foreground; robustness study (absent) |
| Ablations not reproducible from shipped configs | 24-epoch/no-CBGS config not released | — | — |
| Heavy training dependency chain | LiDAR + lidarseg + nuImages masks | Method needs 3 modalities at train | Limits transfer to settings without LiDAR |

---

## 16. Limitations

**Methodological** — cross-view fusion is an **unnormalised sum**, inherited and never examined (§7); no view weighting, no visibility handling; requires accurate calibration and ego-motion; in-box supervision improves size but not orientation (mAOE is its weakest metric); `keep_threshold=0.1` is a hard, unablated foreground cut-off.
**Dataset** — nuScenes only; needs **LiDAR + lidarseg + predicted instance masks** at training; 6-camera driving rig with marginal overlap.
**Computational** — 8 GPUs for training; custom `bev_pool_v2` CUDA op; 17–25 FPS inference; **training time/memory never reported**.
**Generalisation** — no cross-dataset experiment (e.g. Waymo/Argoverse); performance depends on nuImages pre-training (the ★ column matters).
**Implementation** — pinned mmcv 1.5.3 / mmdet 2.27.0; no logs or checkpoints in the folder; `configs/my/` contains 11 undocumented variants (`bevmydistill*`, `geobev-correct-*`) that are **not** referenced in the README — **[INF] leftover research configs that make it ambiguous which config produced which published number**; the shipped configs do not match the stated ablation protocol (§10).

---

## 17. Research gaps

**Gap 1 — Cross-view aggregation is a bare `.sum(1)` and has never been examined in this line of work.**
**[CODE]** `rcsample.py:378`. Every LSS-descendant (BEVDet, BEVDepth, BEVStereo, SA-BEV, GeoBEV) sums over cameras without normalisation or weighting. In overlap regions this scales features by the (geometry-determined) number of observing cameras.
> **Direction:** replace `.sum(1)` with (a) `sum / valid_count`, (b) a learned per-(cell, view) gate, or (c) CVT-style attention — and measure. **[INF] This is a one-line change with a shipped multi-baseline harness (Table 3's protocol) to validate it, and the whole sub-field has left it untouched.** On nuScenes the effect may be small (marginal overlap); the interesting result would be showing it **grows with rig overlap** — testable by combining this with a high-overlap dataset.

**Gap 2 — Per-view confidence exists but is view-local.**
`context_weight = depth_weight ⊙ fg_weight` is computed inside each camera with no cross-view communication. A pixel that two cameras disagree about is treated exactly like one they agree on.
> **Direction:** a **cross-view consistency term** on the depth/foreground scores — if camera *i* and camera *j* both project to the same BEV cell, their predicted occupancy should agree. This is ImGeoNet's (P3) variance idea imported into the LSS family, and it needs **no extra labels**.

**Gap 3 — In-Box supervision improves size but not orientation.**
**[PAPER]** Table 1: best mASE (0.254) but **not** best mAOE (0.318 vs 0.315). The CAI weight `∛(min/max ratios)` is isotropic and rotation-agnostic.
> **Direction:** an **anisotropic / directional** inner weighting — e.g. weight along the box's principal axis, or add an explicit orientation-consistency term on interior pseudo-points. Small, well-scoped, with an obvious metric to move.

**Gap 4 — The radial (polar) intermediate representation is discarded immediately.**
`Bᴿ ∈ ℝ^{C×D×W}` is dense, ego-centric and naturally matches camera geometry — then it is resampled to Cartesian and thrown away.
> **Direction:** run part of the BEV encoder **in radial space** (where resolution naturally matches sensing density: fine near, coarse far), then convert. PolarFormer argues for polar BEV; GeoBEV builds the polar representation and does not exploit it.

**Gap 5 — No missing-camera or camera-failure study.**
Safety-critical setting, six cameras, zero robustness analysis. CVT (P5) does this; GeoBEV does not.
> **Direction:** camera-dropout robustness for LSS-family detectors. **[INF] With an unnormalised sum, dropping a camera should hurt *more* than it does for a normalised/attention fusion — a clean, falsifiable prediction that directly motivates Gap 1.**

---

## 18. What to borrow

| Useful idea | Why it works | Limitation | How to adapt |
|---|---|---|---|
| ⭐ **Supervise the object's *interior*, not its visible surface (In-Box Label)** | Sensor labels record only the ego-facing surface; the *object* is a volume. Filling it teaches size and extent. Worth **+2.2 mAP with zero added parameters** | Needs GT 3D boxes + instance masks; helps size more than orientation | **The most transferable idea in this paper.** Any multi-view method that builds a 3D/BEV representation can supervise "inside the object" rather than "on the surface" — including ImGeoNet's (P3) occupancy head, whose thin-object failures (**picture AP 0.043**) are exactly a surface-vs-volume problem |
| **The three In-Box corrections** (occlusion → unsupervised; background-in-box → masked out; behind-surface → negative) | Turns a crude heuristic into a careful label. Worth **+0.5 mAP** over the vanilla version | Depends on predicted masks | Reusable label-engineering recipe; the "don't supervise what you can't see" principle is general |
| **Centroid-Aware Inner weighting** | Points near the centroid are more reliably interior; weighting by ∛ of face-distance ratios encodes that smoothly. **+0.3 mAP, 0 params** | Isotropic → no orientation signal | Make it **anisotropic** (Gap 3) |
| **Dense gather instead of sparse scatter (RC-Sampling)** | Bilinear sampling from a dense radial map *cannot* leave holes; scatter-pooling can. Fastest **and** most accurate in Table 4 | Adds a resampling step; polar→Cartesian interpolation blurs far field | Generalises to any lift-to-3D pipeline. **[INF] "Gather, don't scatter" is a good default rule** |
| **Contract the height dimension by matmul (Eq. 3)** | Avoids materialising the `C×D×H×W` frustum tensor — >90 % memory saving in the transform stage | — | Directly reusable wherever you build voxel/BEV features from image features |
| **Plug-in validation across 3 baselines (Table 3)** | Independent architectures improving by similar margins is strong evidence the contribution is real, not tuned | — | ⭐ **Copy this protocol.** If your fusion improvement is genuine, demonstrate it on MVDet *and* ImGeoNet *and* CVT — that is far more convincing than one number |
| **Shipping `benchmark_view_transformer.py`** | Makes the efficiency claim independently checkable at the module level | — | **Copy this.** Benchmark your *fusion module* in isolation, not just end-to-end |

---

## 19. Verdict

**GeoBEV is a strong engineering paper whose contribution lies outside your research question.**

- **Do not** treat it as multi-view fusion research: **[CODE]** its cross-view aggregation is a bare `.sum(1)` at `rcsample.py:378`, never discussed or ablated. What it improves is the *per-view lift* and the *depth supervision*.
- **Do not** choose it as a baseline: it has the heaviest setup cost of your six (custom CUDA op, nuScenes + lidarseg + nuImages masks, 8-GPU training, no shipped logs), and its ablation configs are not released.
- **Do** borrow two things: (i) ⭐ the **In-Box Label / interior supervision** principle — parameter-free, +2.2 mAP, and directly applicable to ImGeoNet's thin-object failure mode; (ii) the **"gather, don't scatter"** lifting pattern.
- **Do** copy two *protocols*: the **plug-in-across-three-baselines** validation (Table 3) and the **module-level efficiency benchmark** (Table 4 + `benchmark_view_transformer.py`).
- **Do** note Gap 1 as an easy, credible contribution: an entire sub-field sums camera features without normalisation, and nobody has checked whether that matters.

**Cross-references:** P3 (ImGeoNet) is the indoor analogue that *does* normalise by valid-view count — the direct fix for Gap 1 · P5 (CVT) is the attention alternative to `.sum(1)` · P2 (MSMVD) shows the multi-scale axis GeoBEV addresses only within a view (`scale_num=2`), not across views.
