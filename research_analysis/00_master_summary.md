# Master Summary — Six-Paper Investigation for *Multi-View Object Recognition*

**Author:** research assistant analysis for Siddhartha · **Date:** 2026-08-29
**Scope:** P1–P6 in `Evaluation of codes/` — papers, source code, shipped logs and shipped ground-truth data.

**Evidence tags used everywhere:**
**[PAPER]** what the paper claims · **[CODE]** verified by reading the source · **[LOG]** read from shipped training logs · **[OBS]** measured by me this session · **[INF]** my inference

> ### ⚠ Headline honesty statement
> **No model was executed.** This machine has **no PyTorch and no CUDA GPU** (verified: `import torch` → `ModuleNotFoundError`; `nvidia-smi` → not found), and **zero checkpoints and zero image datasets** exist across all six folders. Steps 5–6 of your plan (run the code, reproduce the results) **could not be performed** and are documented as failures in §4.
> What I did instead: **verified every architectural claim against source**, **mined the one repo that shipped real training logs**, and **ran four original analyses on shipped ground-truth annotations** — no GPU required. Those are in §5 and are the most novel content here.

---

## Where everything is

```
Evaluation of codes/
├── P1/REPORT_P1_MVDet.md                    ← full per-paper analyses
├── P2/REPORT_P2_MSMVD.md                       (Steps 2,3,4,5,6,9,10,11,12,13,16)
├── P3/REPORT_P3_ImGeoNet.md
├── P4/REPORT_P4_CaMuViD.md
├── P5/REPORT_P5_CrossViewTransformers.md
├── P6/REPORT_P6_GeoBEV.md
└── research_analysis/
    ├── 00_master_summary.md                 ← THIS FILE (Steps 1, 20)
    ├── 02_code_analysis/code_trace_tables.md          (Step 4)
    ├── 03_experiment_logs/experiment_log.md           (Step 19)
    ├── 04_results/E1_view_coverage_*.txt              (my original results)
    │             E2_camuvid_ablation_vs_oracle.txt
    │             E3_cross_view_scale_variation.txt
    ├── 05_failure_analysis/failure_synthesis.md       (Step 11)
    │                      E4_imgeonet_per_class_failures.txt
    ├── 06_comparison/master_comparison.md             (Step 14)
    ├── 07_research_gaps/research_gaps.md              (Steps 12,13,15)
    ├── 08_research_directions/research_directions_and_roadmap.md  (Steps 16,17,18)
    └── 09_scripts/*.py                                (re-runnable, CPU only)
```

---

## 1. Inventory (Step 1)

| ID | Paper | Year | Main task | Dataset | Code | Pretrained | Data on disk | Logs | Reproducibility |
|---|---|---|---|---|---|---|---|---|---|
| **P1** | **MVDet** — Multiview Detection with Feature Perspective Transformation (Hou, Zheng, Gould) | ECCV 2020 | MV pedestrian detection → BEV occupancy | Wildtrack, MultiviewX | ✅ | ✗ (link) | ✗ | ✗ | ❌ needs **2 GPUs** (hard-coded), MATLAB, both datasets |
| **P2** | **MSMVD** — Multi-scale Image Features via Multi-scale BEV Features (Yamane et al., NTT) | arXiv 2025 | MV pedestrian detection | GMVD, WT, MVX | ❌ **none** | ✗ | ✗ | ✗ | ❌ **no implementation exists** |
| **P3** | **ImGeoNet** — Image-induced Geometry-aware Voxel Representation (Tu et al., NTHU/Amazon) | ICCV 2023 | **MV 3D object detection, 17–189 classes** | ScanNetV2/200, ARKitScenes | ✅ | ✗ **never released** | splits only | ⭐ **✅ 3 logs, 932 KB** | ❌ torch 1.7.1/mmcv 1.2.7, `nvcc`, Linux, TB-scale data |
| **P4** | **CaMuViD** — Calibration-Free Multi-View Detection (Daryani et al., U. Florida) | CVPR 2025 | MV pedestrian detection → per-view 2D boxes | Wildtrack, MultiviewX | ✅ | ✗ (link) | ⭐ **✅ COCO GT, 25 MB** | ✗ | ❌ DCNv3 CUDA op, no images, 2-stage training |
| **P5** | **CVT** — Cross-view Transformers (Zhou & Krähenbühl, UT Austin) | CVPR 2022 | Map-view semantic segmentation | nuScenes | ✅ | ✗ **never released** | splits only | ✗ | ❌ 60 GB nuScenes. ⭐ **lightest software stack** |
| **P6** | **GeoBEV** — Geometric BEV Representation (Zhang et al., Beihang) | AAAI 2025 | **MV 3D object detection, 10 classes** | nuScenes | ✅ | ✗ (link) | ✗ | ✗ | ❌ CUDA op + lidarseg + nuImages masks + **8 GPUs** |

**Verified totals:** 1529 files, 116 MB · **0** `.pth`/`.ckpt` files · **0** image datasets · 1 repo with training logs · 1 repo with ground-truth annotations · 1 paper with no code.

---

## 2. Relevance to your topic (Step 2) — and an ambiguity you must resolve

**⚠ None of the six papers is multi-view object *classification* in the MVCNN sense** (N rendered views of one isolated object → a category label). All six are multi-view *detection* or *segmentation* of objects in scenes. That matters for how you position your thesis.

| ID | What "multi-view" means | Input | Output | Relevance (a) *many viewpoints → categories* | (b) *many cameras → scene objects* |
|---|---|---|---|---|---|
| P1 | 7/6 fixed calibrated cameras, one scene | N RGB + calib | BEV occupancy, **1 class** | 2 / 5 | **4 / 5** |
| P2 | 6–7 fixed cameras, N varies across GMVD scenes | N RGB + calib | multi-scale BEV occupancy, **1 class** | 2 / 5 | **3.5 / 5** |
| **P3** | ⭐ **Many unordered viewpoints of the same objects** (moving camera), N free: **train 20 / test 50, swept 10–100** | T RGB + K, pose | **3D box + 18/189/17 categories** | ⭐ **4.5 / 5** | 3.5 / 5 |
| P4 | 7/6 fixed cameras, **no calibration** | N RGB | per-view 2D boxes, **1 class** | 2 / 5 | 3 / 5 |
| **P5** | 6 surround cameras, N-agnostic architecture | 6 RGB + calib | binary BEV mask (**no instances**) | 2 / 5 (task) · ⭐ **5 / 5 (mechanism)** | 4 / 5 |
| P6 | 6 surround cameras + 2/8 temporal frames | 6 RGB + calib | **3D box + 10 categories** | 2.5 / 5 | 3 / 5 |

**Net ranking for your purposes: P3 (4/5) ≈ P5 (4/5) > P1 (3.5) > P2 (3) > P4 (2.5) ≈ P6 (2.5).**

> **You need to decide** whether your topic is reading (a) or (b). It changes your baseline, dataset and framing. See §11 and the roadmap's "Decision" box.

---

## 3. Research landscape — how the six differ

**One axis explains almost everything: *where each paper puts its contribution*.**

| Pipeline stage | P1 | P2 | P3 | P4 | P5 | P6 |
|---|---|---|---|---|---|---|
| Per-view features | | ⭐ multi-scale | | ⭐ DCNv3 | | ⭐ multi-scale depth |
| **Image → 3D/BEV lift** | ⭐ feature warp | ⭐ MSP + height slices | ⭐ geometry shaping | ⚠ *no spatial lift* | ⭐ implicit (attention) | ⭐ RC-Sampling |
| **Cross-view fusion** | concat | max | mean+var | concat | ⭐ **attention** | sum |
| Post-fusion reasoning | ⭐ large-kernel | ⭐ BEV-FPN | 3D convs | FRM | bottlenecks | BEV ResNet |
| Supervision design | aux per-view | deep sup. | ⭐ depth→occupancy | cycle-consistency | visibility mask | ⭐ **In-Box + CAI** |
| Recognition head | 1 class | 1 class | ⭐ **18/189** | 1 class ×N | binary mask | ⭐ **10 classes** |

> ### ⭐ The single most important finding of this whole investigation
> **Across six papers spanning 2020–2025, exactly one (P5/CVT) contributes anything to cross-view fusion itself.** The other five innovate on *lifting* or *supervision* and reuse a fixed, parameter-free reduction — concat, max, mean or sum — inherited unchanged from earlier work. Verified line by line:
>
> | Paper | Fusion | Line |
> |---|---|---|
> | P1 | `torch.cat(world_features + [coord_map], dim=1)` | `persp_trans_detector.py:77` |
> | P2 | max-pool along the view direction | **[PAPER]** §3.1 (no code) |
> | P3 | `avg_vol = vol / valid_count` | `imgeonet.py:103-107` |
> | P4 | `torch.cat([f[i] for f in projected], dim=1)` | `Custom_TwoStageDetector.py:716` |
> | **P5** | `rearrange('b n Q K -> b Q (n K)'); softmax(-1)` | ⭐ `encoder.py:157-158` |
> | P6 | `.view(B, N, C, h, w).sum(1)` | `rcsample.py:378` |

---

## 4. Reproduction results (Steps 5–7) — **all attempts failed; here is exactly why**

### Environment (measured)
```
Windows 11 · Python 3.14.3 · conda: base only
PyTorch:  NOT INSTALLED   (ModuleNotFoundError: No module named 'torch')
GPU:      NONE            (nvidia-smi not found)
MATLAB:   not present
```

| Problem | Cause | Attempted | Result |
|---|---|---|---|
| Import torch | Absent from the only env | `conda env list`, `python -c "import torch"` | ❌ |
| Any CUDA execution | No NVIDIA device | `nvidia-smi` | ❌ |
| P1 forward pass | **[CODE]** `.to('cuda:1')` / `.to('cuda:0')` at `persp_trans_detector.py:37,43` | read source | ❌ needs **2** CUDA devices |
| P1 official metric | **[CODE]** `evaluate.py` starts a MATLAB engine | read source | ❌ |
| P3 environment | torch 1.7.1+cu110 / mmcv-full 1.2.7 / mmdet 2.10.0; `rotated_iou` CUDA op | read `0_install_env.sh` | ❌ no Python-3.14 wheels; Linux-first; needs `nvcc` |
| P4 environment | DCNv3 CUDA op (`ops_dcnv3/make.sh`), mmcv-full 1.5.0 | read README | ❌ needs `nvcc` |
| P6 environment | `bev_pool_v2` CUDA op; 8-GPU recipe | read README + source | ❌ |
| P5 environment | pure pip — ⭐ **only stack that would install** | read `requirements.txt` | ⚠ blocked only by data |
| **Datasets** | Wildtrack/MultiviewX/ScanNet/ARKit/nuScenes | recursive scan | ❌ **none present** |
| **Weights** | all six | recursive scan for `*.pth`/`*.ckpt` | ❌ **zero found**; P3 and P5 **never released any** |

### Paper vs. reproduction

| Paper | Metric | Paper reported | My reproduction | Difference | Explanation |
|---|---|---|---|---|---|
| P1 | Wildtrack MODA | 88.2 | **not attempted** | — | no GPU/data/weights/MATLAB |
| P2 | GMVD MODA | 80.2 | **not attempted** | — | **no code exists** |
| P3 | ScanNet mAP@0.25 | 54.8 | not attempted | — | no GPU/data/weights |
| P4 | Wildtrack MODA | 95.0 | not attempted | — | no GPU/images/weights |
| P5 | nuScenes IoU (Set. 2) | 36.0 | not attempted | — | no GPU/nuScenes/weights |
| P6 | nuScenes NDS (test) | 0.662 | not attempted | — | no GPU/data/weights/8 GPUs |

### ⭐ What I *could* verify: P3's paper vs its own shipped logs

| Dataset | Metric | **Paper** | **README** | **[LOG] final** | Δ | Assessment |
|---|---|---|---|---|---|---|
| ScanNetV2 | mAP@0.25 | 54.8 | 54.57 | **54.57** (ep 12/12) | −0.23 | README = log exactly ✔ |
| ScanNetV2 | mAP@0.50 | 28.4 | 28.94 | **28.94** (ep 12/12) | +0.54 | paper's 28.4 = the **epoch-9** value |
| ARKitScenes | mAP@0.25 / @0.50 | 60.2 / 43.4 | 59.82 / 42.76 | **59.82 / 42.76** (ep 12/12) | −0.38 / −0.64 | README = log exactly ✔ |
| ScanNet200 | mAP@0.25 / @0.50 | 22.3 / — | 22.38 / 9.67 | **22.38 / 9.67** | +0.08 | ⚠ **epoch 11 of 30** |

**[INF] Verdict: P3 is an honest release.** README numbers match the logs to two decimals — the authors published actual runs, not cherry-picked bests. Paper figures run 0.2–0.6 higher, consistent with the README's own caveat *"Performance may vary slightly depending on the number of GPUs"* (logs used 4×P40, 1×3090, 3×3090).

**⚠ Three caveats I found that the paper does not state:**
1. **[LOG] ScanNet200 training was abandoned at epoch 11 of 30.** The log ends `Saving checkpoint at 11 epochs` with `eta: 1 day, 4:45:03` remaining; `lr_config step=[8,29]` means **the second LR drop never fired**. The published 22.38 comes from a ~37 %-complete schedule.
2. **[LOG] `scannet.txt` declares `model.type='ImVoxelNet'`** yet logs `loss_occ`/`acc_occ`, and its git hash (`a75dd1a`) differs from the other two (`a5dfdf3`) by 18 months ⇒ produced by a **different code revision** than the release.
3. **[LOG] Run noise is ±0.2 mAP** (ScanNet epochs 9–12: 54.52 / 54.69 / 54.46 / 54.57). Never claim a smaller difference.

---

## 5. ⭐ Original experiments I *could* run (no GPU needed)

All four use only shipped data + pure Python. Scripts in `09_scripts/`, outputs in `04_results/` and `05_failure_analysis/`.

### E1 — View coverage (Wildtrack / MultiviewX ground truth)

| Wildtrack test (1001 person-instances) | Value |
|---|---|
| Mean cameras seeing each person | **5.26 of 7** |
| Per-camera coverage | C6 **96.3 %** · C3 92.8 % · C1 91.9 % · C2 83.4 % · C5 71.9 % · C7 62.2 % · **C4 27.7 %** |
| Greedy coverage order | **C6 → C3 → C1 → C7 → C2 → C4 → C5** |
| Coverage at k = 1,2,3,4 | 96.30 → 98.50 → **99.50** → **100.00 %** |
| Cameras adding **0.00 pp** | **C2, C4, C5** |
| Pairwise Jaccard | C1↔C6 **92.8 %**, C3↔C6 92.0 % |

**MultiviewX:** mean 4.97/6; **zero** single-view instances; **C3+C4 alone reach 100 %**.

### ⭐ E2 — CaMuViD's camera ablation vs. the oracle ceiling

| added camera | Δ oracle coverage | Δ CaMuViD recall | Δ MODA |
|---|---|---|---|
| C4 | **+0.00 pp** | +3.4 pp | +3.2 |
| C5 | **+0.00 pp** | +1.4 pp | +0.0 |
| **C7** | +0.40 pp | **−0.7 pp** | **−0.6** |

**Three findings:**
1. **Cameras 4 and 5 help purely through occlusion disambiguation** — zero coverage gain, +4.8 pp recall. The clearest evidence anywhere in your set that *coverage* and *disambiguation* are separable contributions.
2. **The 7th camera is net-harmful.** CaMuViD's own best is **6 cameras (95.6 MODA)**; its headline is the worse 7-camera number (95.0).
3. **[CODE]** The ablation blanks views to **black images** (`camuvid.py:243-248`) without retraining, and **still scores blanked cameras' GT** (`camuvid.py:328-330`, skip commented out) ⇒ it measures **missing-view robustness**, not view-count scaling.

### E3 — Cross-view scale variation (tests MSMVD's premise, which has no code)

| | Wildtrack | MultiviewX |
|---|---|---|
| Median box-height ratio for the same person across views | **2.73×** | 2.89× |
| ≥ 2× ratio | **90.3 %** | 73.0 % |
| **Would land on ≥2 different FPN levels across views** | **56.5 %** | 30.1 % |

✅ **MSMVD's motivation is confirmed and quantitatively large** — and it is the one paper with no implementation.

### E4 — Failure mining from P3's logs
**[LOG]** ScanNet worst classes: **picture 0.0427**, window 0.2466, curtain 0.3372, door 0.4041 · best: toilet 0.9518, bed 0.8407. ARKit: **tv_monitor 0.0355 (AP@0.50 = 0.0000)**. ScanNet200: **63 of 189 classes (33.3 %) have AP@0.25 = AR@0.25 = 0** — never recalled once; median class AP 0.10 vs mean 0.2238. Mean AP@0.25→AP@0.50 drop **47.0 %**; showercurtain **−95.8 %**.

---

## 6. Performance analysis

| | P1 | P2 | P3 | P4 | P5 | P6 |
|---|---|---|---|---|---|---|
| Params | ~30 M (**19 M in fusion alone at N=7**) | 22.9 M | 485 MB | >60 M (33.6 M in 2 projection FCNs) | ⭐ **5 M** | not reported |
| Latency | ✗ | ❌ never measured | 490 ms/scene @50 views | ✗ | ⭐ **35 FPS** | 17–25 FPS |
| Training memory | ✗ | ✗ | **[LOG]** 10.8–19.4 GB | ✗ | ✗ | ✗ |
| GPUs to train | **2** | ✗ | 1–4 | ~2 | 1–4 | **8** |
| Parameter-matched control | ⭐ ✓ | ⭐ ✓ | ✓ | ✗ | ⭐ ✓ | ⭐ ✓ |

**Is each model accurate because of a genuinely useful multi-view strategy, or because it is bigger?**

| Paper | Answer |
|---|---|
| P1 | ⭐ **Strategy.** Same net, 3 projection choices: 26.8 → 68.2 → **88.2** MODA at ~equal compute |
| P2 | ⭐ **Strategy.** MSMVD-R18 (22.9 M, 80.2) beats baseline-R101 (43.2 M, 75.8) |
| P3 | ⭐ **Strategy.** +18 % size, +18 % time for **+6.1/+4.6 mAP**; the MVS-depth alternative costs **15.7×** runtime for *less* gain |
| P4 | ⚠ **Largely capacity + protocol.** Fusion does no geometric work (§7 of its report); gains plausibly from InternImage + N Cascade R-CNN heads + an image-space metric |
| P5 | ⭐⭐ **Strategy, most convincingly.** *Smallest* (5 M) and *fastest* (35 FPS) while matching FIERY — with the **same backbone and decoder** |
| P6 | ⭐ **Supervision.** In-Box + CAI worth **+2.2 mAP with zero added parameters**; gains transfer as a plug-in to 3 independent baselines |

---

## 7. Failure analysis (summary — full version in `05_failure_analysis/`)

| Failure | Evidence | Cause | Fix |
|---|---|---|---|
| **Thin / flat / wall-flush objects** | picture 0.043, tv_monitor 0.036 **[LOG]** | thinner than a 0.16 m voxel; surface probability can't separate object from wall | interior (In-Box) supervision, finer/anisotropic voxels |
| **Localisation collapse @ IoU 0.5** | mean −47.0 %; showercurtain −95.8 % **[LOG]** | **[CODE]** nearest-neighbour back-projection `.round().long()` | trilinear sampling, box refinement |
| **Long tail never recalled** | 63/189 classes at AP=AR=0 **[LOG]** | resolution + class imbalance | not a fusion problem — report median AP too |
| ⭐ **Redundant view hurts** | 7 cams < 6 cams **[PAPER]**; C7 adds 0.40 pp **[OBS]** | uniform fusion cannot down-weight | **learned gating + monotonicity constraint** |
| ⭐ **Coverage ≠ disambiguation** | C4/C5: +4.8 pp recall at 0.00 pp coverage **[OBS]** | no corroboration signal in fusion | **set statistics (variance, valid-count)** |
| **Cross-view scale mismatch** | 56.5 % span ≥2 FPN levels **[OBS]** | single-scale or independently-fused scales | multi-scale projection |
| **Far-field decay** | IoU → 0 by 60 m **[PAPER]** | few features, weak priors | distance-aware loss, temporal fusion |
| **Out-of-FoV ≡ empty** | **[CODE]** `warp_perspective` zero-pads | no validity mask | explicit validity input (P3 has it; P1, P6 do not) |

---

## 8. Limitations across all six

**Methodological** — fixed N (P1, P4); order-dependent fusion (P1, P4); no view weighting (all but P5); no validity normalisation (all but P3); no view-quality estimation (all six); flat-ground assumption (P1); no spatial alignment at all (P4).
**Dataset** — 400-frame near-saturated benchmarks (P1, P2, P4); **[OBS]** massively over-covered rigs; single object class (P1, P2, P4); no instances (P5); heavy extra-modality training requirements (P3 depth, P6 LiDAR + lidarseg + masks).
**Computational** — 2 GPUs (P1), 8 GPUs (P6), custom CUDA ops (P3, P4, P6), cost never measured (P2, P4).
**Generalisation** — only P4 reports cross-dataset transfer; only P2 tests an unseen camera layout (GMVD); nobody tests unseen viewpoint *distributions*.
**Implementation / evaluation** — **0 checkpoints on disk**; P3 & P5 never released any; P2 has no code; deprecated pinned stacks throughout; **[CODE]** P1's **silent** MATLAB→Python evaluator fallback (repo's own README: *0–2 MODA lower*); **[CODE]** P4's image-space IoU protocol tabulated beside BEV-protocol competitors; **[LOG]** P3's ScanNet200 number from an unconverged run.

---

## 9. Common research gaps (full version in `07_research_gaps/`)

1. ⭐ **Cross-view fusion is a fixed, unlearned reduction in 5 of 6 papers** — despite **[OBS]** cameras differing by 3.5× in coverage and 2.7× in observation scale.
2. ⭐ **"More views is better" is assumed and is demonstrably false** — **[PAPER]** 7 cameras < 6 cameras; **[PAPER]** views 50→100 buy only +2.2 mAP.
3. ⭐ **Coverage and disambiguation are conflated** — the only signal that separates them (cross-view agreement) is never computed, or computed and discarded (**[CODE]** P3 keeps variance out of the detector).
4. **Missing-view robustness is barely studied**, and the one fixed-N study **blanks** views rather than removing them.
5. **View-count curves are compared across rigs of incomparable redundancy** — the curve's *shape* is set by the rig, not the method, and no paper reports a redundancy statistic.
6. **Multi-scale is handled within views but not across them** — only P2, which has no code.
7. **Fixed-N architectures are still being published in 2025** (P4).
8. **Evaluation protocols have fragmented** and results are tabulated as if they had not.
9. **Recognition is thin** — 4 of 6 papers predict a single class; where 189 classes are tested, **[LOG]** a third are never recalled.
10. **Nothing is reproducible end-to-end** from what is on disk.

---

## 10. Ideas most relevant to your research

| Rank | Idea | Source | Evidence it works |
|---|---|---|---|
| 1 | ⭐ **Softmax attention over the union of (views × patches)** — permutation-invariant, N-agnostic **and** a learned per-cell view selector | P5 | 5 M params @ 35 FPS matches a 7 M @ 8 FPS competitor with the **same backbone and decoder**; camera-aware embedding worth **+5.0 IoU** |
| 2 | ⭐ **Masked mean + cross-view variance, normalised by `valid_count`** | P3 | Enables **train-20 / test-50 views**; variance drives geometry shaping for **+26 ms constant** vs 15.7× for an MVS cascade |
| 3 | ⭐ **Multi-scale projection** — BEV features inherit their source image scale | P2 | **+3.1 MODA** alone; **[OBS]** 56.5 % of pedestrians span ≥2 FPN levels across views |
| 4 | ⭐ **Interior (In-Box) supervision** instead of surface supervision | P6 | **+2.2 mAP with zero added parameters**; directly targets **[LOG]** P3's thin-object failures |
| 5 | **Calibration as a positional embedding**, not a warp | P5 | +5.0 IoU; and a *learned per-camera code* recovers 34.4 of 36.0 → a semi-calibration-free path |
| 6 | **Project features, not pixels or detections** | P1 | 88.2 vs 68.2 vs 26.8 MODA, same network |
| 7 | **Large receptive field after fusion** | P1, P2, P4 | +11.3 / +5.3 / +0.4 — consistent across three independent papers |
| 8 | **Auxiliary supervision from a train-only modality** | P3, P6 | Geometric priors without a deployment sensor |

**Protocols to copy:** parameter-matched backbone sweep (P2) · plug-in validation on ≥2 baselines (P6) · params+FPS+GPU-hours beside accuracy (P5) · matched ablation configs (P3, P5) · ship your annotations (P4).

---

## 11. Recommended research direction

> ### The target
> **A multi-view aggregation operator that is permutation-invariant and N-agnostic, predicts a per-(location, view) informativeness weight from content and geometry, preserves cross-view agreement statistics rather than normalising them away, and operates scale-by-scale.**

| | Learned view weighting | Cross-view agreement | Multi-scale through fusion |
|---|:-:|:-:|:-:|
| P1 · P4 · P6 | ✗ | ✗ | ✗ |
| P2 | ✗ | ✗ | ⭐ ✓ |
| P3 | ✗ | ⚠ computed, **discarded** | ✗ |
| P5 | ⭐ ✓ | ✗ (softmax removes it) | ⚠ partial |
| **Your target** | ⭐ ✓ | ⭐ ✓ | ⭐ ✓ |

**[INF] No paper in this set occupies more than one cell.** Each component is independently validated by evidence above; the combination is unoccupied. 🔍 *Verify against MVCNN/GVCNN/view-GCN/MVTN, set-transformer/deep-sets, and next-best-view literature before claiming novelty.*

---

## 12. Proposed baseline

| Role | Choice | Why |
|---|---|---|
| ⭐ **Primary** | **P3 ImGeoNet** | Only genuinely N-agnostic, permutation-invariant repo (**[CODE]** train-20/test-50); ships **matched ablation configs** and **three real logs**; publishes the **views-vs-accuracy curve**; predicts **categories**. ⚠ No weights → must train. **Start with ARKitScenes**: **[LOG]** proven on **1 × RTX 3090**, 12 epochs, ~13 h, 19.4 GB |
| ⭐ **Architectural reference** | **P5 CVT** | The only learned fusion; lightest stack (pure pip, **no CUDA ops**, 1 GPU, `overfit.py`); `cameras` config makes view experiments free |
| Secondary | **P1 MVDet** | Canonical baseline; `--variant` ablations runnable. ⚠ 2 GPUs + MATLAB |
| Idea source only | **P2 MSMVD** | Best idea, no code |
| Critique / evidence source | **P4 CaMuViD** | ⚠ **Not** a numeric baseline (protocol incomparable), but its ablation + my E2 is your strongest gap evidence |
| Less relevant | **P6 GeoBEV** | Contribution is lifting + supervision, not fusion; heaviest setup. Borrow **In-Box Label** only |

---

## 13. Proposed improvement (concrete)

**Step 1 — Set-statistics fusion.** Replace the single reduction with
`z = concat( mean_t(V_t), var_t(V_t), max_t(V_t), valid_count/N )`.
Permutation-invariant, N-agnostic, no new hyper-parameters. **[CODE]** ImGeoNet already computes mean and variance at `imgeonet.py:103-110` and forwards only `avg_vols * occ` at `L142/L167` — a ~5-line change with a shipped matched baseline.

**Step 2 — Monotonic gated aggregation.** Predict `g_t ∈ [0,1]` per (location, view) from the projected feature + validity mask + geometric footprint; fuse as `Σ g_t V_t / Σ g_t`; train with a subset-consistency penalty `max(0, loss(S') − loss(S))` for `S ⊂ S'`.

> **Pre-registered success criterion:** on Wildtrack, **7-camera performance ≥ 6-camera performance**, at or above **95.6 MODA**. No published method satisfies this, and the counter-example is published by the method being improved.

---

## 14. Experimental plan

| Phase | Weeks | Work | Gate |
|---|---|---|---|
| **0** | 1–2 | Rig-redundancy protocol (**no GPU** — scripts already work); secure GPU; download ARKitScenes and/or nuScenes; decide reading (a)/(b)/(c) | Redundancy stats for 3 datasets |
| **1** | 3–6 | Reproduce **ImGeoNet on ARKit** (target 59.8/42.8, log-verified) and/or **CVT** (target 36.0 IoU) | Within **±0.6** of published |
| **2** | 7–10 | View-count sweep (E10/E6) · view-**order** control (must be bit-identical) · attention-mass analysis · missing-view study in **all three** regimes (blank / mask / retrain) | Curves + the redundancy comparison |
| **3** | 11–14 | **Set-statistics fusion** vs matched baseline; ablate each statistic; width-matched control | Gain > ±0.2 run noise |
| **4** | 15–22 | **Monotonic gated aggregation**; + view selection if time | Pre-registered criterion met |
| **5** | 23–28 | Plug-in validation on ≥2 baselines; full ablation; params/FLOPs/FPS table; failure analysis; write-up | Paper draft |

---

## 15. Potential contributions *(label as potential until validated)*

1. **Quantitative characterisation of view redundancy in standard benchmarks** — *partly complete already* (E1–E3).
2. **First explicit separation of coverage from disambiguation** in multi-view detection, with evidence that views 4–5 give +4.8 pp recall at zero coverage gain.
3. **Set-statistics multi-view aggregation** preserving cross-view agreement.
4. **Monotonic view aggregation** with a pre-registered "adding a view never hurts" guarantee.
5. **Complementarity-based view selection** — ~2× speed-up at equal accuracy on a published saturating curve.
6. **Rig-redundancy-normalised evaluation protocol**.
7. **Reproducibility findings** *(already established)*: P3's ScanNet200 number from an epoch-11-of-30 run; P1's silent evaluator fallback; P4's non-comparable protocol.

---

## 16. Answer to your final question

> *After deeply studying these six works, what have we learned, what are the limitations, and what direction can I pursue?*

**What we learned.** Multi-view recognition has advanced almost entirely by improving **how a single view is lifted into a shared space** (perspective warping → geometry-aware voxels → radial BEV sampling → attention-based implicit lifting) and **how that lifting is supervised** (auxiliary per-view heads → depth-induced occupancy → interior In-Box labels). Aggregating the views themselves has barely moved since 2020: five of six papers still use concat, max, mean or sum.

**What the limitations are.** Views are treated as interchangeable and equally reliable. The data says otherwise: **[OBS]** cameras differ by 3.5× in how many objects they see and 2.7× in the scale at which they see them; three of Wildtrack's seven cameras add literally nothing to coverage; and **[PAPER]** a published method's seventh camera *reduces* its own accuracy. Meanwhile, the one signal that would let a model tell a corroborating view from a redundant one — cross-view agreement — is either never computed or, in the single case where it is, deliberately routed away from the recogniser.

**The direction.** Make view aggregation **content-adaptive, agreement-aware and scale-aware**, and hold it to a standard nobody currently meets: **adding a view must never make the result worse.** That claim is falsifiable, has a published counter-example to beat, and sits in a gap that all six of these papers leave open.

---

### Two things to decide before Phase 1
1. **Which reading of "Multi-View Object Recognition"** is yours — (a) many viewpoints → categories, (b) many cameras → scene objects, or (c) both? This sets your baseline and dataset (§2, roadmap Decision box).
2. **What GPU access you will have.** It determines whether you start with P5/CVT (1 GPU, easy stack, 60 GB download) or P3/ImGeoNet (1 GPU proven for ARKit, hard legacy stack, but the better task fit).
