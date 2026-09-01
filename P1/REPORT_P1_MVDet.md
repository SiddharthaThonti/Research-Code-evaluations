# P1 — MVDet: Multiview Detection with Feature Perspective Transformation

**Analysis date:** 2026-08-29
**Analyst context:** M.Tech research on *Multi-View Object Recognition*
**Evidence base:** paper PDF (`P_1_2007.07247v2_M.pdf`, 16 pp.) + full source tree (`MVDet/`)
**Execution status:** ❌ not executed — see §9. No results in this report were produced by running this code.

Throughout, statements are tagged:
**[PAPER]** = claimed in the paper · **[CODE]** = verified by reading the source ·
**[OBS]** = measured by me in this session · **[INF]** = my inference/interpretation

---

## 1. Inventory

| Item | Value |
|---|---|
| Title | Multiview Detection with Feature Perspective Transformation |
| Authors | Yunzhong Hou, Liang Zheng, Stephen Gould (ANU / Australian Centre for Robotic Vision) |
| Venue / Year | **ECCV 2020** (arXiv v2, 1 May 2021) |
| Code | `MVDet/` — present, complete, git history intact (`master`) |
| README | Yes (3.5 KB) |
| Datasets | Wildtrack (real), MultiviewX (synthetic, released with paper) — **neither present on disk** |
| Pretrained weights | Linked to OneDrive in README — **not present on disk** |
| Config files | None (argparse flags in `main.py` only) |
| Train script | `main.py`, `run_gpu01.sh` |
| Eval script | `multiview_detector/evaluation/evaluate.py` (+ MATLAB devkit, + Python fallback) |
| Requirements | `requirements.txt` (torch 2.4.0, kornia 0.7.3, numpy 1.24.4) |
| Result files | None. `gt-demo.txt` / `test-demo.txt` are **evaluator self-test fixtures**, not model outputs |
| Reproducibility | **Blocked** (no data, no weights, needs 2 GPUs + MATLAB) |

---

## 2. Relevance to *Multi-View Object Recognition*

**A. Problem solved.** Detect the ground-plane positions (occupancy) of pedestrians in a crowded scene observed by *N* synchronised, calibrated, fixed cameras with overlapping fields of view.

**B. What "multi-view" means here.** **Multiple fixed calibrated cameras** viewing one shared scene from different static positions. It is *not* multiple viewpoints of a single isolated object, and *not* sequential/active view selection. *N* is fixed per dataset (7 for Wildtrack, 6 for MultiviewX).

**C. Input.** `imgs ∈ ℝ^{B×N×3×720×1280}` — N synchronised RGB frames, plus per-camera intrinsics `A` and extrinsics `[R|t]` loaded from XML. **[CODE]** `frameDataset.__getitem__` returns `(imgs, map_gt, imgs_gt, frame)`.

**D. Output.** A single-channel **pedestrian occupancy map (POM)** `g̃ ∈ ℝ^{H_g×W_g}` on the ground plane (120×360 for Wildtrack, 160×250 for MultiviewX; 1 cell = 10 cm). Auxiliary output: per-view head/foot heatmaps (2 channels per view). **No object category is predicted** — single class, "pedestrian".

**E. Relevance score.**

| Reading of your topic | Score | Why |
|---|---|---|
| *Multi-camera recognition of objects in a scene* | **4 / 5** | Directly the same problem family; MVDet is the canonical baseline every later paper (P2, P4) compares against |
| *Multi-viewpoint recognition of a single object (MVCNN-style)* | **2 / 5** | Task differs (localisation, not classification); but the **anchor-free projected-feature representation** transfers |

**Net: 3.5 / 5.** Take it as the *reference baseline and reference failure case* for view fusion, not as an architecture to copy.

---

## 3. Research problem, formally

- **Input:** `{I_n}ⁿ₌₁ᴺ`, `I_n ∈ ℝ^{3×H_i×W_i}`; calibrations `θ⁽ⁿ⁾ = A⁽ⁿ⁾[R|t]⁽ⁿ⁾`.
- **Output:** `g̃ ∈ ℝ^{H_g×W_g}`, occupancy probability per ground cell.
- **Assumptions:** (i) cameras synchronised; (ii) calibration known and accurate; (iii) all pedestrians stand on `z = 0`; (iv) *N* fixed and identical at train and test.
- **Constraints:** memory — 1080×1920 downsampled to 720×1280; batch size 1.
- **Objective:** minimise MSE against a Gaussian-smoothed occupancy target.

**Two sub-problems the paper names** **[PAPER]**:
1. **Multiview aggregation** — how to combine cues from N views at one ground location.
2. **Spatial aggregation** — how to combine information from neighbouring ground locations.

---

## 4. Motivation and research gap addressed

**[PAPER]** Prior multi-view detectors represented a ground location by **anchor-box features** (ROI-pooled from a box computed from an assumed 1.8 m × 0.6 m human). If the person is sitting, crouching or unusually sized, that box mostly contains background → corrupted representation (paper Fig. 3, the seated woman in white). Second, prior work used **CRF / mean-field inference** for spatial reasoning — machinery outside the CNN, needing hand-designed potentials.

**The gap MVDet targets:** replace *inaccurate anchor-box aggregation* with **anchor-free per-pixel feature sampling**, and replace *CRF* with **large-kernel dilated convolution**, giving one end-to-end differentiable network.

---

## 5. Previous approaches (as grouped by the paper)

| Category | Examples | What it does | Limitation MVDet names |
|---|---|---|---|
| Detection-result fusion | RCNN&clustering, POM-CNN | Detect per view, then cluster/mean-field the 2D results | Single-view detections are already destroyed by occlusion; almost no information left to fuse (11.3 / 23.2 MODA) |
| Anchor-box feature fusion | DeepMCD, Deep-Occlusion | ROI-pool a projected human-sized box per view, fuse | Anchor boxes are inaccurate → representation contaminated by background |
| CRF / mean-field spatial reasoning | Deep-Occlusion, Roig et al. | Model ground-plane neighbour consistency | Needs hand-designed potentials + non-CNN inference |
| Geometric transforms in DL | STN, perspective transform | Learn/apply parametric warps | Not previously applied to multi-view feature aggregation for detection |

---

## 6. Proposed method

```
  I₁ ─┐
  I₂ ─┤   ResNet-18 (shared weights, last 3 strides → dilation)
  ... ─┼──►  C=512 feature map  ──►  bilinear resize to 270×480
  I_N ─┘                                     │
                                             ├──► img_classifier (1×1,1×1) → head/foot heatmaps  [aux loss]
                                             │
                                    perspective warp with θ⁽ⁿ⁾  (kornia warp_perspective)
                                             │
                            ┌────────────────┴────────────────┐
                            │  N projected maps, each C×H_g×W_g │
                            └────────────────┬────────────────┘
                                             │  CHANNEL CONCAT  +  2-channel coord map
                                             ▼
                                   (N·512 + 2) × H_g × W_g
                                             │
                     map_classifier: Conv3×3(→512) → Conv3×3 d=2(→512) → Conv3×3 d=4(→1)
                                             ▼
                                  occupancy map g̃  (H_g × W_g)
```

### Module-by-module **[CODE]** `multiview_detector/models/persp_trans_detector.py`

| Module | In | Out | Operation | Why needed |
|---|---|---|---|---|
| `base_pt1`/`base_pt2` | `3×720×1280` | `512×90×160` | ResNet-18, last 3 strides replaced by dilation (`replace_stride_with_dilation=[False,True,True]`) → 8× downsample instead of 32× | Keeps spatial resolution high enough that a warped pixel still localises a person |
| `F.interpolate` | `512×90×160` | `512×270×480` | bilinear upsample to `H_f,W_f` | Fixed feature raster so one precomputed homography works |
| `proj_mats[cam]` | — | `3×3` | `map_zoom · (permutation · (A[R|t]_{z=0})⁻¹) · img_zoom`, precomputed in `get_imgcoord2worldgrid_matrices` | Closed-form image→ground homography; **not learned** |
| `warp_perspective` | `512×270×480` | `512×H_g×W_g` | kornia inverse-warp, zero-pad outside FoV | The "feature perspective transformation" |
| `create_coord_map` | — | `2×H_g×W_g` | normalised X-Y grid (CoordConv) | Convolutions are translation-invariant; the ground plane is not (cameras sit at fixed places) |
| `map_classifier` | `(N·512+2)×H_g×W_g` | `1×H_g×W_g` | 3 convs, dilations 1/2/4 → **15×15-cell (1.5 m × 1.5 m) receptive field** | The CRF replacement — "spatial aggregation" |
| `img_classifier` | `512×270×480` | `2×270×480` | 1×1 → 1×1 | Auxiliary head/foot supervision per view |

**Key implementation detail the paper does not stress** **[CODE]**: the backbone is **split across two GPUs** — `base_pt1` on `cuda:1`, `base_pt2` and everything else on `cuda:0` (lines 37–38, 43–44). Two CUDA devices are mandatory, not optional.

---

## 7. Multi-view fusion strategy — the part that matters most for you

**Mechanism: CHANNEL CONCATENATION of perspective-warped feature maps.** **[CODE]** line 77:

```python
world_features = torch.cat(world_features + [coord_map], dim=1)   # (N·C + 2)
```

**Why it helps** **[PAPER]** — three deliberate choices, each validated by the paper's own ablation (Table 3):

| Choice | Alternative | MODA (Wildtrack) |
|---|---|---|
| project **feature maps** | project RGB pixels | 88.2 vs **26.8** |
| project **feature maps** | project detection results | 88.2 vs **68.2** |
| **+ large-kernel** spatial agg. | no large kernel | 88.2 vs **76.9** |

Projecting *features* wins because a 512-d feature vector already encodes its own spatial neighbourhood, so it survives the geometric structure break that warping causes; raw pixels do not.

### Weaknesses of concat fusion — **[INF]**, all confirmed in code

1. **Architecture is hard-wired to N.** `nn.Conv2d(out_channel * self.num_cam + 2, 512, 3, ...)` (line 51). Change *N* → different first-layer shape → **must retrain from scratch**. You cannot evaluate MVDet with 3 views using a 7-view checkpoint.
2. **Not permutation-invariant.** View *n* always occupies channels `[n·512, (n+1)·512)`. Camera identity is baked into the weights. Shuffling the camera order at test time is out-of-distribution. (In this dataset the order is constant, so this is latent, not visible in the paper's numbers.)
3. **No view weighting whatsoever.** Every view contributes exactly 512 channels regardless of whether the person is 10 px or 1000 px tall in it, occluded, or outside the FoV. **Out-of-FoV regions are zero-padded** — a zero block and a "person definitely absent" block are indistinguishable to the first conv.
4. **Cost grows linearly in N.** First conv has `(512N+2)×512×3×3` weights ≈ **16.5 M params at N=7** — about 1.5× the whole ResNet-18 backbone (11.2 M). Fusion, not perception, dominates the parameter budget.
5. **Single scale.** One feature level (1/8 resolution). **[OBS]** In the Wildtrack test annotations the *same person* spans a **median 2.73× box-height ratio across cameras** (90.3 % of people ≥ 2×, p95 = 5.59×) — see `research_analysis/04_results/E3_*`. A single-scale feature map is structurally mismatched to that. This is precisely the gap P2/MSMVD attacks.

---

## 8. Loss function **[CODE]** `loss/gaussian_mse.py`, `trainer.py:44-47`

```
L_ground     = ‖ g̃ − f(g) ‖₂                                        (Eq. 3)
L_single⁽ⁿ⁾  = ‖ s̃_head⁽ⁿ⁾ − f(s_head⁽ⁿ⁾) ‖₂ + ‖ s̃_foot⁽ⁿ⁾ − f(s_foot⁽ⁿ⁾) ‖₂   (Eq. 4)
L_combined   = L_ground + α · (1/N) Σₙ L_single⁽ⁿ⁾                   (Eq. 5),  α = 1
```

`f(·)` = Gaussian smoothing of the sparse GT. **[CODE]** implemented as `adaptive_max_pool2d` to the prediction resolution, then `conv2d` with a fixed Gaussian kernel (σ = 20/grid_reduce = 5 for the map, kernel 41×41).

**[INF]** MSE on a Gaussian heatmap, not focal/BCE. The target is ~0 almost everywhere, so the loss is dominated by background; this is why the threshold `cls_thres=0.4` and the NMS radius (20 cells = 0.5 m) matter so much at inference.

**Ablation the paper reports (Fig. 8):** removing the single-view loss (α=0) costs only **−1.2 MODA** on Wildtrack and **−2.0** on MultiviewX. **[INF]** The auxiliary loss is nearly redundant — foot points are already supervised through the ground plane. The paper says the *head* branch is the part that helps, because it supplies information the ground-plane loss cannot.

---

## 9. Datasets

| | Wildtrack | MultiviewX |
|---|---|---|
| Type | Real | Synthetic (Unity + PersonX models) |
| Cameras | 7 | 6 |
| Area | 12 × 36 m | 16 × 25 m |
| Grid | 480 × 1440 (2.5 cm) → 120 × 360 after `grid_reduce=4` | 640 × 1000 → 160 × 250 |
| Resolution | 1080×1920 → resized 720×1280 | same |
| Frames | 400 @ 2 fps | 400 @ 2 fps |
| Crowdedness | 20 person/frame | 40 person/frame (configurable 20–80) |
| Avg camera coverage per location | 3.74 | 4.41 |
| Split | first 90 % train / last 10 % test **[CODE]** `train_ratio=0.9` | same |
| Augmentation | **None** — only Resize + ToTensor + Normalize **[CODE]** `main.py:40` | same |
| Validation set | **None** — the test set is evaluated every epoch and the final checkpoint is kept | same |

**[OBS] My measurement of Wildtrack test GT** (from the annotations shipped in `P4/CaMuViD/data`, which are the same Wildtrack labels):
- mean **5.26 of 7** cameras see each pedestrian; 55 % are seen by ≥ 6 cameras.
- **C1+C3+C6 already cover 99.5 %** of all pedestrian instances; C2, C4, C5 add **0.00 pp** of coverage.
- C4 alone sees only **27.7 %** of instances — 3.5× weaker than C6 (96.3 %).

**[INF] Is this dataset representative of your problem?** Partially. It gives you real occlusion and real calibration, but: one object class, one fixed camera rig, 400 frames, no viewpoint variation between train and test (the *same* 7 cameras in the *same* positions). Any conclusion about *viewpoint generalisation* cannot be drawn from it. GMVD (used by P2) exists precisely to fix this.

---

## 10. Experimental setup **[PAPER]** §4.2 + **[CODE]** `main.py`

| | Paper | Code default | Match? |
|---|---|---|---|
| GPU | 2 × RTX 2080 Ti | requires `cuda:0` **and** `cuda:1` | ✔ (2 GPUs mandatory) |
| Backbone | ResNet-18, dilated | `resnet18(replace_stride_with_dilation=[False,True,True])` | ✔ |
| Input | 720 × 1280 | `T.Resize([720,1280])` | ✔ |
| Feature | C=512, 270×480 | `img_reduce=4` → 270×480 | ✔ |
| Optimiser | SGD, momentum 0.5, wd 5e-4 | same | ✔ |
| LR | OneCycle, max 0.1 | `OneCycleLR(max_lr=0.1)` | ✔ |
| Epochs | 10 | `--epochs 10` | ✔ |
| Batch | 1 | `-b 1` | ✔ |
| α (single-view loss) | 1 | `--alpha 1.0` | ✔ |
| Threshold / NMS | 0.4 / 0.5 m | `cls_thres=0.4`, `nms(..., 20, inf)` (20 cells × 2.5 cm × 4 = 0.5 m ✔) | ✔ |
| Seed | not stated | `--seed 1`, `cudnn.benchmark=True` | ⚠ non-deterministic despite the seed |

**Verdict: paper and code agree on every stated hyper-parameter.** That is unusually clean.

---

## 11. Attempt to execute (Step 5) — **failed, documented**

Environment measured this session:

```
OS         Windows 11 (win32)
Python     3.14.3   (conda base only)
PyTorch    NOT INSTALLED  (ModuleNotFoundError: No module named 'torch')
CUDA GPU   NONE          (nvidia-smi not found)
MATLAB     not present
```

| Problem | Cause | Solution attempted | Result |
|---|---|---|---|
| Cannot import torch | No PyTorch in the only conda env | Checked `conda env list`, `python -c "import torch"` | ❌ Confirmed absent |
| Cannot run any CUDA code | No NVIDIA GPU / driver on this machine | `nvidia-smi` | ❌ Not found |
| Model requires **2** CUDA devices | Hard-coded `.to('cuda:1')` / `.to('cuda:0')` in `persp_trans_detector.py:37,43` | Read source | ❌ Would need code edit even with 1 GPU |
| No dataset | `~/Data/Wildtrack`, `~/Data/MultiviewX` expected; searched whole tree | `Get-ChildItem -Recurse` for `Wildtrack`/`MultiviewX` dirs | ❌ Only P4's COCO JSONs exist; no images |
| No pretrained weights | README links OneDrive | Searched tree for `*.pth` | ❌ Zero `.pth` files anywhere |
| Official metric needs MATLAB | `evaluate.py` starts a MATLAB engine | Read source | ❌ Not installed |

**Nothing was run. No MVDet number in this report is a reproduction.**

### ⚠ Reproducibility trap I found in the code **[CODE]** `evaluation/evaluate.py:21-33`

```python
def evaluate(...):
    try:
        import matlab.engine
        ...                      # official MOTChallenge devkit
    except:                      # <-- bare except, silent
        from ...pyeval.evaluateDetection import evaluateDetection_py
        ...                      # unofficial Python re-implementation
```

A bare `except:` silently swaps the evaluator. The repo's own `pyeval/README.md` states the Python version gives **"approximately 0~2 % decrease in MODA, MODP"** and says *"please use the official MATLAB API if you want to obtain the same evaluation result shown in the paper."*

**Consequence:** anyone without MATLAB reproduces MVDet and gets ~86–88 MODA instead of 88.2, with **no warning printed**. If you use MVDet as your baseline, you must state which evaluator you used, and use the same one for your own model.

---

## 12. Results: paper-reported (not reproduced)

**Wildtrack / MultiviewX, paper Table 3** — reproduced here for reference only.

| Method | WT MODA | WT MODP | WT Prec | WT Rec | MVX MODA | MVX MODP | MVX Prec | MVX Rec |
|---|---|---|---|---|---|---|---|---|
| RCNN & clustering | 11.3 | 18.4 | 68 | 43 | 18.7* | 46.4* | 63.5* | 43.9* |
| POM-CNN | 23.2 | 30.5 | 75 | 55 | – | – | – | – |
| DeepMCD | 67.8 | 64.2 | 85 | 82 | 70.0* | 73.0* | 85.7* | 83.3* |
| Deep-Occlusion | 74.1 | 53.8 | **95** | 80 | 75.2* | 54.7* | **97.8*** | 80.2* |
| MVDet (project images) | 26.8 | 45.6 | 84.2 | 33.0 | 19.5 | 51.0 | 84.4 | 24.0 |
| MVDet (project results) | 68.2 | 71.9 | 85.9 | 81.2 | 73.2 | 79.7 | 87.6 | 85.0 |
| MVDet (w/o large kernel) | 76.9 | 71.6 | 84.5 | 93.5 | 77.2 | 76.3 | 89.5 | 85.9 |
| **MVDet** | **88.2** | **75.7** | 94.7 | **93.6** | **83.9** | **79.6** | 96.8 | **86.7** |

\* re-implemented by the MVDet authors.

| Metric | Paper reported | My reproduction | Difference | Explanation |
|---|---|---|---|---|
| Wildtrack MODA | 88.2 | **not attempted** | — | No GPU, no dataset, no weights (§11) |
| Wildtrack MODP | 75.7 | not attempted | — | " |
| MultiviewX MODA | 83.9 | not attempted | — | " |

**[INF] Consistency check I *can* do without running anything:** the paper's own Fig. 5 caption reports 26.8 / 88.2 / 68.2 for the three projection choices, and Table 3 reports exactly those numbers. The four MVDet variants in Table 3 correspond 1:1 to the four `--variant` options in `main.py` (`img_proj`, `res_proj`, `no_joint_conv`, `default`). **The ablations in the paper are all runnable from the released code** — this is a well-released repo. `res_proj` (ResNet-feature projection variant) is the one variant with no matching row in Table 3.

---

## 13. Controlled experiments — what the repo *would* support

| Experiment | Supported out of the box? | What would need to change |
|---|---|---|
| **A. Number of views** | ❌ | `map_classifier` in-channels `= 512N+2` is fixed. Needs retraining per N, or replacing concat with a pooling/attention fusion |
| **B. View selection (which subset)** | ❌ same reason | Same. **[OBS]** But you can already rank views *without a model*: on Wildtrack test, greedy coverage order is **C6 → C3 → C1 → C7 → C2 → C4 → C5**, saturating at 4 cameras (see §9) |
| **C. View order** | ⚠ possible | Permuting the camera loop in `forward` is a 1-line change and *should* change results, because channels are view-specific. **This is an untested, publishable sensitivity check.** |
| **D. Missing views** | ⚠ partial | Feeding a black image for view *n* keeps the tensor shape valid — this is exactly what P4 does. No code change needed |
| **E. View quality (blur/noise/res)** | ✔ | `train_trans` in `main.py:40` is a plain `T.Compose`; inserting corruptions is trivial |
| **Crowdedness / occlusion sweep** | ✔ (data-side) | MultiviewX is regenerable at 20/40/60/80 person-per-frame — the paper already did this (Fig. 7: MODA 79.9 → 64.4 as crowdedness goes 20 → 80) |

---

## 14. Ablations already in the repository

The paper's Table 3 ablations are all first-class `--variant` choices **[CODE]** `main.py:58-67`:

| `--variant` | File | What it removes |
|---|---|---|
| `default` | `persp_trans_detector.py` | full MVDet |
| `img_proj` | `image_proj_variant.py` | projects RGB pixels instead of features |
| `res_proj` | `res_proj_variant.py` | projects an intermediate ResNet stage |
| `no_joint_conv` | `no_joint_conv_variant.py` | removes the large-kernel spatial aggregation |

`--alpha 0` reproduces the Fig. 8 single-view-loss ablation. **[INF]** This is a genuinely reproducible ablation suite — better than most repos.

---

## 15. Performance / cost analysis **[INF]**, computed analytically from `[CODE]`

| Quantity | Value | Derivation |
|---|---|---|
| Backbone params | ~11.2 M | ResNet-18 minus fc/avgpool, **shared across all N views** |
| `img_classifier` | 512·64 + 64·2 ≈ 33 K | 1×1 convs |
| `map_classifier` conv1 @ N=7 | (512·7+2)·512·3·3 ≈ **16.5 M** | `(512N+2)×512×3×3` |
| `map_classifier` conv2 | 512·512·9 ≈ 2.36 M | |
| `map_classifier` conv3 | 512·1·9 ≈ 4.6 K | |
| **Fusion head total @ N=7** | **≈ 18.9 M** | **1.7× the backbone** |
| **Fusion head @ N=3** | ≈ 4.7 M | scales linearly in N |
| Peak activation | N × 512 × 120 × 360 floats @ N=7 ≈ **619 MB** in fp32 for the concatenated tensor alone | why batch size is 1 |

**Answer to "is it accurate because of a good multi-view strategy or because it's big?"**
**[INF] Genuinely the strategy.** The paper's own controls settle it: the *same* backbone and *same* head, changing only *what* gets projected, moves MODA from 26.8 → 68.2 → 88.2. Compute is essentially unchanged across those three; only the representation changes. However, the *large-kernel head* (+11.3 MODA) does buy its gain with ~19 M extra parameters, which is a real cost the paper does not price.

---

## 16. Failure analysis

The repo contains **no saved predictions**, so this is derived from the paper's own reported numbers plus my data measurements — labelled accordingly.

| Failure type | Evidence | Probable cause | Potential fix |
|---|---|---|---|
| Precision below CRF methods (94.7 vs 95.0 for Deep-Occlusion) | **[PAPER]** Table 3 | MSE-on-Gaussian + fixed 0.4 threshold produces diffuse blobs; NMS at 0.5 m merges/splits imperfectly | Focal loss + learned offset regression (this is exactly what MVDeTr and P2/MSMVD later added) |
| Degrades sharply with crowding: 79.9 → 64.4 MODA as MultiviewX goes 20 → 80 person/frame | **[PAPER]** Fig. 7 | Ground-plane cells become ambiguous; concat fusion has no mechanism to say "this view is occluded here" | Per-view, per-cell confidence/visibility weighting |
| Spatial aggregation helps far less on MultiviewX (+6.7) than Wildtrack (+11.3) | **[PAPER]** §4.4 | MultiviewX has 4.41 vs 3.74 cameras per location — more coverage → less ambiguity to resolve | Confirms the gain comes from *disambiguation*, not coverage |
| Anchor-free gain smaller on MultiviewX (+7.2) than Wildtrack (+9.1) | **[PAPER]** §4.4 | Synthetic humans are near-identical in size, so anchor boxes are accurate there | Real-world size variance is where the idea pays |
| **[OBS]** Scale mismatch across views ignored | Median **2.73×** cross-view box-height ratio on Wildtrack; **56.5 %** of people would land on *different* FPN levels in different views | Single-scale feature map | Multi-scale projection (P2's MSP) |
| **[OBS]** Redundant views carry equal weight | C2/C4/C5 add 0.00 pp coverage; C4 sees only 27.7 % of people yet contributes 512/3586 channels always | No view weighting | Learned view gating (P5's cross-attention does this implicitly) |
| Out-of-FoV = zeros, indistinguishable from "empty" | **[CODE]** `warp_perspective` zero-pads | No validity mask input to `map_classifier` | Concatenate a per-view validity mask — ImGeoNet (P3) does exactly this with `valids` |

---

## 17. Limitations

**Methodological**
- Fixed *N*, fixed camera positions, architecture tied to both.
- Requires accurate calibration; the homography is precomputed once and never adapted.
- **Flat-ground assumption** (`z=0`); a single homography per camera. SHOT (a follow-up) needed stacked homographies at multiple heights precisely because of this.
- No view-quality estimation, no view selection, no attention.
- Not permutation-invariant.
- Single-scale features.

**Dataset**
- 400 frames per dataset; test = last 40 frames, temporally adjacent to training frames → optimistic.
- One object class.
- No validation split; the test set is used for model selection every epoch **[CODE]** `main.py:105-121`.
- Zero data augmentation.

**Computational**
- **Requires 2 GPUs** — a hard barrier for you.
- Batch size 1.
- Fusion head is 1.7× the backbone at N=7.

**Generalisation**
- Cannot transfer to a new camera rig without retraining (later work — GMVD, and P4 — reports MVDet drops to **17.0 MODA** cross-dataset).

**Implementation**
- Hard-coded `~/Data/` paths.
- Hard-coded `cuda:0` / `cuda:1`.
- `requirements.txt` pins torch 2.4.0, but `kornia.geometry.transform.warp_perspective`'s default `align_corners` changed across kornia versions — a silent numerical risk when reproducing a 2020 paper with 2024 libraries.
- Silent MATLAB→Python evaluator fallback (§11).

---

## 18. Research gaps — specific, not generic

**Gap 1 — Fusion capacity is spent on view identity rather than view content.**
Concatenation dedicates a fixed 512-channel slot to each camera, so the model must learn "what camera 4 usually looks like" instead of "how informative camera 4 is *right now*". **[OBS]** On Wildtrack, camera C4 observes only 27.7 % of pedestrians yet always occupies 1/7 of the fusion input, while C6 observes 96.3 % and gets the same share.
> **Direction:** replace channel-concat with a **content-conditioned view-weighting** operator that is permutation-invariant and N-agnostic — e.g. a per-(cell, view) scalar gate predicted from the projected feature and a validity mask, followed by weighted pooling.

**Gap 2 — Zero-padding conflates "outside field of view" with "nothing here".**
**[CODE]** `warp_perspective` writes 0 outside the FoV, and no validity mask is passed to `map_classifier`. The network must infer FoV boundaries from the CoordConv map alone.
> **Direction:** propagate an explicit per-view validity/visibility mask into fusion and normalise by the number of *valid* views per cell — exactly what ImGeoNet does (`valid_count`) and MVDet does not.

**Gap 3 — Single-scale projection under large cross-view scale variation.**
**[OBS]** 90.3 % of Wildtrack pedestrians have ≥ 2× height ratio across the views that see them; 56.5 % would be assigned to different FPN levels in different views.
> **Direction:** project multiple feature scales and fuse scale-by-scale (P2/MSMVD's answer; still unavailable as code, so it is open ground for you).

**Gap 4 — View-order dependence is never measured.**
Because fusion is order-dependent by construction, MVDet has an untested invariance property. No paper in this set reports it.
> **Direction:** a cheap, publishable robustness study: permute camera order at test time and report the spread. Any order-dependent fusion (MVDet, CaMuViD) should be penalised for it.

---

## 19. What to borrow for your research

| Useful idea | Why it works | Limitation as implemented | How to adapt |
|---|---|---|---|
| **Project *features*, not pixels or detections** | A deep feature vector already summarises its receptive field, so it survives the spatial-structure break that warping causes. Worth **+61.4 MODA** over pixels and **+20.0** over detections | Uses one fixed scale and one fixed plane | Keep the principle; project **multiple scales** and, if your objects are not on a plane, multiple height slices or a voxel grid |
| **Anchor-free per-location sampling** | Removes the assumed-object-size prior that corrupts ROI features. Worth **+9.1 MODA** over anchor-based DeepMCD | Only valid because a homography gives exact correspondences | Directly transferable to any calibrated setting |
| **CoordConv on the fused map** | Restores absolute position, which plain convs discard, and the ground plane genuinely is not translation-invariant | 2 channels only | Cheap; keep it |
| **Large-kernel dilated convs instead of CRF** | Gets CRF-like neighbourhood reasoning inside the network, end-to-end. Worth **+11.3 MODA** | ~19 M params | Consider dilated **separable** convs, or a small BEV transformer, for the same receptive field at lower cost |
| **Auxiliary per-view supervision** | Forces each view's features to remain individually discriminative | Worth only ~1–2 MODA; the foot branch is redundant | Keep the **head-point** branch only, or replace with a per-view *visibility* prediction that also feeds the fusion gate |
| **Releasing a synthetic twin dataset (MultiviewX)** | Lets you control crowdedness/occlusion — the paper's most informative experiment (Fig. 7) | Synthetic humans are unrealistically uniform in size | Do the same for your setting: a controllable simulator is how you will isolate *view count* from *view quality* |

**What NOT to copy:** the concat fusion itself, the 2-GPU split, and the no-validation-split protocol.

---

## 20. Verdict

MVDet is the **right primary baseline** for the multi-camera branch of your work: the code is complete and faithful to the paper, the ablations are all runnable, and its single design weakness — *uniform, order-dependent, N-fixed concatenation* — is exactly the axis your research can improve. Its practical costs for you are the **2-GPU requirement** and the **MATLAB evaluator**; budget for both, or plan to modify the fusion to a pooling form (which incidentally removes the 2-GPU need, since the fusion head shrinks).

**Cross-references:** P2 (MSMVD) fixes its scale limitation · P4 (CaMuViD) removes its calibration dependence · P5 (CVT) replaces its fusion with attention.
