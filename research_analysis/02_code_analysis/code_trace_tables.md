# Code Execution Traces (Step 4)

For each repository, the actual execution path from input to metric, with the file, class/function and line numbers that determine the research method. Only the parts that *define* the method are listed — not every file.

**[CODE]** = verified by reading the source in this session.

---

## P1 — MVDet (`P1/MVDet/`)

```
main.py
  └─ frameDataset(Wildtrack|MultiviewX, grid_reduce=4)
       └─ __getitem__ → (imgs[N,3,720,1280], map_gt, imgs_gt[N], frame)
  └─ PerspTransDetector(train_set, 'resnet18')
       └─ forward(imgs) → (map_result, imgs_result)
  └─ PerspectiveTrainer.train / .test
       └─ GaussianMSE  →  nms  →  evaluate()  →  MODA/MODP
```

| Stage | File | Function / Class | What it does |
|---|---|---|---|
| Data loading | `multiview_detector/datasets/frameDataset.py` | `frameDataset.__getitem__` (L124-149) | Loads N images per frame, stacks to `[N,3,H,W]`; builds sparse occupancy `coo_matrix` and per-view head/foot maps |
| Split | same | `__init__` (L26-29) | `train_ratio=0.9` — **first 90 % train, last 10 % test; no validation split** |
| Calibration | `datasets/Wildtrack.py` / `MultiviewX.py` | `intrinsic_matrices`, `extrinsic_matrices`, `worldgrid2worldcoord_mat` | Parses camera XMLs |
| Preprocessing | `main.py` L40 | `T.Compose([Resize([720,1280]), ToTensor(), Normalize])` | ⚠ **no augmentation** |
| Feature extraction | `models/persp_trans_detector.py` | `base_pt1` (**`cuda:1`**) → `base_pt2` (**`cuda:0`**) L37-44, L63-64 | ResNet-18, last 3 strides → dilation; ⚠ **two GPUs mandatory** |
| Per-view head | same | `img_classifier` L49-50, L66 | 1×1 convs → 2-ch head/foot heatmap |
| Projection matrices | same | `get_imgcoord2worldgrid_matrices` L89-101 | `perm · (A[R\|t]_{z=0})⁻¹`, precomputed once, **not learned** |
| View processing | same | `warp_perspective(img_feature, proj_mat, grid)` L69 | kornia inverse warp to ground plane; ⚠ **zero-pads outside FoV** |
| **View fusion** | same | **L77 `torch.cat(world_features + [coord_map], dim=1)`** | **Channel concat → `(512N+2)` channels.** Order-dependent, N-fixed |
| Spatial aggregation | same | `map_classifier` L51-54, L81 | Conv3×3 → Conv3×3(d=2) → Conv3×3(d=4); RF = 15×15 cells = 1.5 m |
| Loss | `loss/gaussian_mse.py` + `trainer.py` L44-47 | `GaussianMSE` | `MSE(pred, Gauss(GT))`; `L = L_ground + α·mean(L_single)` |
| Post-proc | `utils/nms.py`, `trainer.py` L100-155 | threshold `cls_thres=0.4`, `nms(..., 20, inf)` | 20 cells × 2.5 cm × grid_reduce 4 = **0.5 m** |
| **Evaluation** | `evaluation/evaluate.py` L21-33 | `evaluate()` | ⚠ **bare `except:` silently falls back** from MATLAB devkit to `pyeval` (repo's own README: *"approximately 0~2 % decrease in MODA, MODP"*) |

**Ablation variants (all runnable via `--variant`)**: `image_proj_variant.py`, `res_proj_variant.py`, `no_joint_conv_variant.py`.

---

## P2 — MSMVD

❌ **No code exists.** Nothing to trace. All method details in the P2 report are **[PAPER]**-only.

---

## P3 — ImGeoNet (`P3/ImGeoNet/mmdetection3d/`)

```
tools/train.py  ← configs/imgeonet/imgeonet_{scannet|scannet200_vx808032|arkit}.py
  └─ ScanNetMultiViewDataset / ARKitDataset
       └─ MultiViewPipeline(n_images=20 train / 50 test, sample_method)
  └─ ImGeoNet.forward_train(img, img_metas, gt_bboxes_3d, gt_labels_3d, depth_maps, depth_masks)
       └─ build_volume → occ_head → extract_feat → bbox_head
  └─ mAP@0.25 / mAP@0.50
```

| Stage | File | Function / Class | What it does |
|---|---|---|---|
| **View sampling** | `mmdet3d/datasets/pipelines/multi_view.py` | `MultiViewPipeline.__call__` L16-28 | `random` → `np.random.choice(ids, n_images)`; `linear` → `np.linspace(0, len-1, n_images)`. Then `for i in sorted(ids)` |
| Preprocessing | config `train_pipeline` | `LoadImageFromFile → Resize(640,480) → Normalize → Pad(480,640)` | ScanNet only; ARKit is already 192×256 |
| Depth (train only) | config | `LoadDepthMap(depth_shift=1000.)` | ⚠ **ARKit test pipeline collects `keys=['img']` only — paper's "images-only at inference" claim ✔ verified** |
| Augmentation | config | `RandomShiftOrigin(std=(.7,.7,.0))` | The **only** augmentation |
| Feature extraction | `models/detectors/imgeonet.py` | `build_volume` L67-76 | ResNet-50 (frozen stage 1, `norm_eval`) + FPN; **`assert stride == 4`** |
| Back-projection | same | `backproject()` L284-303 | `volume[i,:,valid[i]] = features[i,:,y,x]` with `.round().long()` — ⚠ **nearest-neighbour, no bilinear** |
| **View fusion** | same | **L99-110** | `valid_count = valid.sum(0)`; **`avg_vol = vol / valid_count`** (Eq. 4); `var_vol = E[X²] − E[X]²` (Eq. 8); invalid voxels zeroed |
| Geometry shaping | `models/dense_heads/` `OccupancyHead` | called at L134 / L160 | Input `cat(avg_vols, var_vols)` = 512 ch → 3D Enc-Dec → Linear → Sigmoid → `S` |
| Occupancy target | `imgeonet.py` | `compute_target_occ_single` L245-281 | Voxel positive if projected depth matches GT depth within `margin = voxel_z · depth_cast_margin/2` = 0.32 m |
| **Gating** | same | **L142 / L167 `x = self.extract_feat(avg_vols * occ)`** | ⚠ **Only the MEAN is gated and forwarded. `var_vols` never reaches the detector** |
| 3D neck | `models/necks/imvoxelnet.py` | `FastIndoorImVoxelNeck(256→128, n_blocks=[1,1,1])` | Multi-scale 3D convs, L=3 |
| Detection head | `models/dense_heads/imvoxel_head_v2.py` | `ScanNetImVoxelHeadV2` / `SunRgbdImVoxelHeadV2` | Anchor-free FCOS-style; `limit=27`, `centerness_topk=18` |
| Loss | config | Focal(cls) + CE(centerness) + AxisAlignedIoU/Rotated-IoU(box) + **Focal(occ) weight 10** | All other weights = 1 |
| Evaluation | mmdet3d `indoor_eval` | `--eval mAP` | mAP@0.25 / mAP@0.50, per class |
| **Matched baseline** | `configs/imgeonet/imvoxelnet_*.py` | — | ⭐ Verified by diff: differs **only** by `occ_head`, `depth_cast_margin`, and depth loading |

---

## P4 — CaMuViD (`P4/CaMuViD/`)

```
camuvid.py  ← config.py (Config_d)
  └─ JointDetectDataset(view_pairs=combinations, num_views=num_head, blank_views)
  └─ CustomTwoStageDetector(..., num_head=7)
       └─ extract_feat(imgs) → (features, post_projected, projected, A^p, A^b)
  └─ evaluation.py: evaluate_detection_with_hungarian(iou_threshold=0.45)
```

| Stage | File | Function / Class | What it does |
|---|---|---|---|
| Config | `config.py` | `Config_d` | ⚠ `num_head = 7`, `num_views = 7` **hard-coded**; `mode = "train"|"test"`; `blank_views = None` |
| Data loading | `custom_datasets_fn.py` | `JointDetectDataset.__getitem__` L124-140 | Builds anchor + positive-pair views |
| **View blanking** | same L132-137 **and** `camuvid.py` L243-248 | — | ⚠ `Image.new('RGB', size, (0,0,0))` — **removed cameras get a BLACK IMAGE**; model is **not retrained** |
| ⚠ GT of blanked views | `camuvid.py` L328-330 | — | `# if view in blank_views: continue` is **commented out** → blanked cameras' GT **is still scored** |
| Preprocessing | `config.py` L66-81 | `ResizeKeepRatio((640,1333))` → `ToTensor` → `Normalize` → `PadToSizeDivisibleBy(32)` | |
| Feature extraction | `Custom_TwoStageDetector.py` | InternImage-T backbone + FPN | COCO-pretrained; DCNv3 custom CUDA op |
| Projection net | same L382-416 | `ProjectionMatrixNetwork` | `GAP(F)` → 256→128→256→**65 536** → reshape **256×256** |
| Module creation | same L499-525 | — | ⚠ per-view `rpn_head_{i+1}`, `roi_head_{i+1}`, `refiningmodule_{i}`; but `for i in range(2)` for `projection_net_{1,2}` — **two shared nets** |
| **"Projection"** | same **L681-683** | `torch.matmul(A^p, F.view(B,C,H*W))` | ⚠ **Channel-mixing applied identically at every spatial location. NO spatial warp** |
| **View fusion** | same **L716-719** | `torch.cat([f[i] for f in projected], dim=1)` → `Conv2d(256·N, 256, k=1)` | ⚠ Concatenates **pixel (h,w) of camera 1 with pixel (h,w) of camera 2** — geometrically unrelated; the 1×1 conv has no spatial context to repair it |
| Back-projection | same L741 | `torch.matmul(A^b, P_f.view(B,C,H*W))` | Per view |
| Refinement | same L745 | `refiningmodule_i` — Conv7×7 → ReLU → Conv3×3 → ReLU | Per view |
| Detection | same | `roi_head_{i+1}` (Cascade R-CNN) | **N independent detectors** |
| Loss | `camuvid.py` / paper Eq. 7 | Cascade R-CNN losses + `1e-4 · Σ‖A^b·P_i − F_i‖₁` | ⚠ **1e-4** makes the invertibility constraint near-vacuous; never ablated |
| **Evaluation** | `evaluation.py` L43-116 | `evaluate_detection_with_hungarian(iou_threshold=0.45)` | ⚠ **2D image-space box IoU + Hungarian**, per-identity TP across views; **not** the BEV 0.5 m protocol used by every competitor in its Table 1 |
| FP handling | same L145-170 | `project_unmatched_boxes_to_world` | Unmatched predictions projected to world coords and clustered |

---

## P5 — CVT (`P5/cross_view_transformers/`)

```
scripts/train.py  ← config/config.yaml + experiment/cvt_nuscenes_{vehicle|road}.yaml
  └─ NuScenesGeneratedDataset (cameras=[[0,1,2,3,4,5]])
  └─ CrossViewTransformer = Encoder + Decoder
       └─ Encoder.forward(batch) → BEV latent (128×25×25)
       └─ Decoder → 200×200 logits
  └─ MultipleLoss(BinarySegmentation | Center) ; IoUMetric
```

| Stage | File | Function / Class | What it does |
|---|---|---|---|
| **Rig definition** | `config/data/nuscenes.yaml` L10 | `cameras: [[0,1,2,3,4,5]]` | ⭐ A **list of camera rigs**. Changing it changes N with **zero code edits** |
| Data loading | `data/nuscenes_dataset.py` | `NuScenesDataset.parse_scene` L134-163, `CAMERAS` L111-112 | `for cam_idx in camera_rig` — builds one sample per rig |
| Preprocessing | `config/data/nuscenes.yaml` L28-31 | `h:224, w:480, top_crop:46`; `augment:'none'` | |
| Feature extraction | `model/backbones/efficientnet.py` | `EfficientNetExtractor` | EfficientNet-B4, taps `reduction_2` (28×60) + `reduction_4` (14×30) |
| BEV query | `model/encoder.py` L66-111 | `BEVEmbedding` | Learned `nn.Parameter(σ·randn(128,25,25))` + ego-frame world grid via `V⁻¹` |
| Camera embedding | same L218, L244-246 | `cam_embed(E_inv[...,-1:])` | Camera **centre** in world coords → 128-d |
| Image embedding | same L217, L248-253 | `img_embed(E_inv · K_inv · pixel)` | **Ray direction** → 128-d |
| Key / Value | same L255-256, L267, L271 | `key = normalize(d−c) + feature_proj(φ)`; `val = feature_linear(φ)` | Geometry **+** appearance |
| Query | same L258-262, L274 | `query = normalize(w_embed − c_embed) + x` | Per-(camera, BEV cell) |
| ⭐ **View fusion** | same **L156-161** | `dot = einsum('b n Q d, b n K d -> b n Q K')`; `rearrange('b n Q K -> b Q (n K)')`; **`softmax(dim=-1)`** | **One softmax over ALL views × ALL patches** → permutation-invariant, N-agnostic, **learned per-cell view weighting** |
| Refinement | same L312, L331-335 | 2 × `ResNetBottleNeck` after each of 2 CVA blocks | |
| Decoder | `model/decoder.py` | 3 × (bilinear ×2 + conv), `residual: True` | 25×25 → 200×200 |
| Loss | `losses.py` L27-79 | `BinarySegmentationLoss` / `CenterLoss` = `sigmoid_focal_loss(α=−1, γ=2)` | ⚠ `min_visibility` masks low-visibility cells **out of the loss** (L52-54) |
| Metric | `metrics.py` L39-72 | `IoUMetric(thresholds=[0.4,0.5])` | ⚠ Same `min_visibility` mask applied (L65-71) — **reported IoU is over visible cells only** |
| Ablation flag | `model/encoder.py` L208-214 | `no_image_features: bool` | Table 3's "no image features in keys" row, as a config flag |
| Unused | same L49-63 | `RandomCos` | Table 4's Random-Fourier row is implemented but **not wired to any config** |

---

## P6 — GeoBEV (`P6/GeoBEV/`)

```
tools/train.py  ← configs/geobev/geobev-r50-nuimage-cbgs.py
  └─ CBGSDataset(NuScenesDatasetBEVDet, img_info_prototype='bevdet4d')
  └─ GeoBEV(BEVDepth4D)
       └─ extract_img_feat → RCSample.forward → bev_encoder → CenterHead
  └─ mAP / NDS
```

| Stage | File | Function / Class | What it does |
|---|---|---|---|
| Data | config | `PrepareImageInputsGeoBEV`, `LoadAnnotationsBEVDepth` | `input_size=(256,704)`, `Ncams`, CBGS class-balanced sampling |
| Extra labels | `tools/generate_point_label.py` | — | ⚠ Requires **nuScenes-lidarseg**; produces depth + semantic labels |
| Extra masks | README §2b | `samples_instance_mask` | ⚠ HTC-on-nuImages predictions, downloaded separately |
| Backbone | config | ResNet-50, `load_from='ckpts/nuimage_pretrained_r50.pth'` | ⚠ **nuImages-pretrained, not ImageNet** |
| Neck | config | `CustomFPN(out_channels=512)` | |
| Depth + foreground | `models/necks/rcsample.py` L41-112, L345-358 | `DepthNet` | `D=118` bins over [1,60] m @0.5 m, **+1 foreground channel**; calibration-conditioned via `mlp_input`; `scale_num=2` refines at `downsamples=[16,8]` |
| Gating | same L360-365 | `depth_weight = where(softmax(D) ≥ 1/D, sigmoid(D), 0)`; `fg_weight = sigmoid(fg) ≥ 0.1` | Per-pixel confidence — **within one view only** |
| **Radial BEV (Eq. 3)** | same **L366-367** | `torch.matmul(context_weight.permute(0,3,1,2), context.permute(0,3,2,1))` | Contracts the **height** dim → `ℝ^{C×D×W}`; avoids materialising the 4D frustum |
| **Cartesian (Eq. 4)** | same **L377** | `F.grid_sample(frustum_feat.view(B*N,C,D,W), norm_bev_coor)` | **Gather**, not scatter → no feature vacancy |
| **View fusion** | same **L378** | **`bev_feat.view(B, N, out_ch, h, w).sum(1)`** | ⚠ **Unnormalised SUM over the 6 cameras. The entire cross-view fusion.** No `valid_count`, no weighting |
| Temporal fusion | `models/detectors/geobev.py` L142 | `torch.cat(bev_feat_list, dim=1)` | Concat over T frames (`multi_adj_frame_id_cfg`) |
| In-Box Label | same L165-230 | builds `gt_bboxes_inbox`; returns `loss_fg`, `loss_inbox` | Supervises the object **interior**; 3 corrections (occlusion / background-in-box / behind-surface) |
| BEV encoder | config | `CustomResNet` (160/320/640) + `FPN_LSS` → 256 | |
| Head | config | `CenterHead` + `CenterPointBBoxCoder`, rotate-NMS | 10 classes + velocity + attributes |
| Loss | config | `GaussianFocal(6.0)` + `L1(1.5)` + `BCEFocal` depth `[300,600]` + `BCEFocal` fg `[33,67]` + **CAI** | |
| Cost tools | `tools/analysis_tools/` | `get_flops.py`, `benchmark.py`, **`benchmark_view_transformer.py`** | ⭐ Module-level efficiency benchmarking shipped |
| ⚠ Config gap | `configs/geobev/` | `max_epochs=20` **with** CBGS | Paper's ablations use **24 epochs without CBGS** — **no such config is shipped** |
| ⚠ Stray configs | `configs/my/` | 11 files (`bevmydistill*`, `geobev-correct-*`) | Undocumented; not referenced in README |

---

## Cross-cutting observations

1. **Only P5 makes fusion a learned operation.** P1/P4 concat, P2 max, P3 mean, P6 sum — all fixed, parameter-free reductions applied identically at every location.
2. **Only P3 normalises by the number of valid views** (`/valid_count`). P1 zero-pads, P6 sums unnormalised, P2 maxes, P4 has no notion of validity.
3. **P1 and P4 are the only architectures hard-wired to N** (`Conv2d(C·N + 2, …)` and `Conv2d(256·N, 256, 1)` + N replicated heads). P2, P3, P5, P6 are N-agnostic by construction.
4. **Three papers ship first-class ablation switches**: P1 (`--variant`), P3 (matched `imvoxelnet_*.py` configs), P5 (`no_image_features`, `cameras`). P4 and P6 do not.
5. **Two evaluation traps**: P1's silent MATLAB→Python evaluator fallback (0–2 MODA), and P4's image-space IoU protocol tabulated against BEV-protocol competitors.
