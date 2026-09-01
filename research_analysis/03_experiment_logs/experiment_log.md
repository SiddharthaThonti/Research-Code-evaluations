# Experiment Log

**Rule: never overwrite. Append new entries; keep failed attempts.**

Session 1 = 2026-08-29, analysis session on the local Windows machine.

---

## Environment record — 2026-08-29 (baseline for all entries below)

```
OS              Windows 11 Home Single Language 10.0.26200 (win32)
Shell           PowerShell 5.1 / Git Bash
Python          3.14.3
Conda envs      base only  (C:\Users\Siddhartha\anaconda3)
PyTorch         NOT INSTALLED   -> ModuleNotFoundError: No module named 'torch'
CUDA / GPU      NONE            -> nvidia-smi not found
MATLAB          not present
Repo size       1529 files, 115.96 MB across P1..P6
.pth/.ckpt      0 files found anywhere
Datasets        0 image datasets present
```

**Consequence: E0 below is the controlling result — no repository in this project can be executed on this machine.** Everything else in this log is either static code analysis or an analysis of data/logs that shipped with the repos.

---

## E0 — Environment feasibility check (all six repos)

| Field | Value |
|---|---|
| **Experiment ID** | E0 |
| **Paper** | P1–P6 |
| **Date** | 2026-08-29 |
| **Goal** | Determine whether any repo can be executed as released |
| **Commands** | `python --version`; `conda env list`; `nvidia-smi --query-gpu=...`; `python -c "import torch"`; recursive scans for `*.pth`, dataset dirs |
| **Hardware** | see Environment record |
| **Runtime** | < 1 min |
| **Result** | ❌ **All six blocked.** No PyTorch, no CUDA GPU, no datasets, no checkpoints |
| **Per-repo blockers** | **P1**: needs 2 CUDA devices (`cuda:0`+`cuda:1` hard-coded, `persp_trans_detector.py:37,43`), MATLAB engine for the official metric, Wildtrack+MultiviewX. **P2**: no code exists at all. **P3**: torch 1.7.1+cu110 / mmcv-full 1.2.7 / mmdet 2.10.0 (Linux-first, no Python-3.14 wheels), `nvcc` for `rotated_iou` op, TB-scale ScanNet frame extraction, **no checkpoints ever released**. **P4**: DCNv3 custom CUDA op (`ops_dcnv3/make.sh`), mmcv-full 1.5.0 + mmdet 2.28.1, images absent, two-stage training needing a backbone checkpoint that is also absent. **P5**: 60 GB nuScenes + 361 MB label archive, **no checkpoints ever released**; software stack itself is clean. **P6**: `bev_pool_v2` CUDA op, nuScenes + **lidarseg** + **nuImages instance masks**, 8-GPU training recipe |
| **Expected result** | At least pretrained-model evaluation on one repo |
| **Difference** | Total block |
| **Observations** | P5 has by far the lightest software stack (pure pip, no custom CUDA ops, single GPU trainable, `scripts/overfit.py` for sanity checks). Its only real barrier is the dataset download. P6 has the heaviest (2 custom-op builds, 3 data modalities, 4 separate downloads, 8 GPUs) |
| **Conclusion** | Steps 5–6 (execution, reproduction) cannot be completed here. Pivot to (a) static code verification, (b) mining shipped logs, (c) original analyses of shipped ground-truth data |
| **Next** | E1 |

---

## E1 — View coverage & informativeness from ground-truth annotations

| Field | Value |
|---|---|
| **Experiment ID** | E1 |
| **Paper** | P4 (data source); informs P1, P2, P4 |
| **Date** | 2026-08-29 |
| **Code version** | `research_analysis/09_scripts/view_coverage.py` (written this session) |
| **Dataset** | `P4/CaMuViD/data/{Wildtrack,MultiviewX}/*_coco_test_anno_chunk_id.json` — shipped ground truth, no images needed |
| **Configuration** | Pure Python 3.14, `json` + `itertools` only. No model, no GPU |
| **Command** | `python research_analysis/09_scripts/view_coverage.py` |
| **Hardware** | CPU only |
| **Runtime** | ~40 s (exhaustive over all 2⁷−1 camera subsets) |
| **Output** | `research_analysis/04_results/E1_view_coverage_wildtrack_multiviewx.txt` |
| **Result — Wildtrack test** | 40 frames, 1001 (frame,person) instances, 5276 boxes. **Mean 5.26 of 7 cameras** see each person. Per-camera coverage: C6 96.3 %, C3 92.8 %, C1 91.9 %, C2 83.4 %, C5 71.9 %, C7 62.2 %, **C4 27.7 %**. Greedy order **C6→C3→C1→C7→C2→C4→C5**: 96.30 → 98.50 → **99.50** → **100.00** → 100.00 → 100.00 → 100.00 %. **C2, C4, C5 each add 0.00 pp.** Pairwise Jaccard C1↔C6 = 92.8 %, C3↔C6 = 92.0 % |
| **Result — MultiviewX test** | 1494 instances, mean 4.97 of 6. **Zero** single-view instances. **C3+C4 alone reach 100 %** coverage |
| **Expected result** | Wildtrack paper states 3.74 cameras cover each ground-plane *cell*; I measure per-*person* visibility, so a higher number (5.26) is consistent |
| **Observations** | Both benchmarks are heavily over-covered for *coverage*. Any accuracy gain from the 4th–7th camera must therefore come from **occlusion disambiguation**, not from seeing new people. Sanity check passed: `annotations/instance = 5.271` (≈ mean coverage), confirming the `_id` field really is a shared person ID |
| **Conclusion** | Redundancy in these rigs is extreme and unequal. Camera C4 contributes 3.5× less coverage than C6 yet receives an identical 1/7 share of MVDet's and CaMuViD's fusion input |
| **Next** | E2 |

---

## E2 — CaMuViD's camera-elimination ablation vs. the data's oracle ceiling

| Field | Value |
|---|---|
| **Experiment ID** | E2 |
| **Paper** | P4 |
| **Date** | 2026-08-29 |
| **Code version** | `research_analysis/09_scripts/camuvid_ablation_check.py` |
| **Dataset** | Wildtrack + MultiviewX shipped COCO GT (test **and** train splits) |
| **Method** | CaMuViD's TP rule is per-identity (*"If an ID is assigned to at least one detection, it is counted as a TP"*, paper §4.2), so max achievable recall for a kept subset S is `oracle(S) = |{(frame,pid) visible in ≥1 camera of S}| / |all|`. Compared against the published Table 3 |
| **Command** | `python research_analysis/09_scripts/camuvid_ablation_check.py` |
| **Runtime** | ~15 s |
| **Output** | `research_analysis/04_results/E2_camuvid_ablation_vs_oracle.txt` |
| **Result** | k=1..7 (prefix C1..Ck): oracle **91.91 / 93.31 / 98.50 / 98.50 / 98.50 / 99.60 / 100.00 %** vs CaMuViD recall **60.2 / 77.9 / 92.1 / 95.5 / 96.9 / 99.3 / 98.6 %**. Achieved/oracle: 65.5 → 83.5 → 93.5 → 97.0 → 98.4 → **99.7** → 98.6 % |
| **Key finding 1** | **C4 and C5 add exactly 0.00 pp of oracle coverage**, yet CaMuViD recall rises **+3.4** and **+1.4 pp**. ⇒ Direct evidence that views 4–5 help purely through **occlusion disambiguation**, not coverage |
| **Key finding 2** | **C7 adds +0.40 pp of coverage but recall drops −0.7 pp and MODA drops −0.6** (95.6 → 95.0). The 7th camera is **net-harmful**. Cross-check from E1: C7 has the largest mean box area (40 595 px) and only 62.2 % coverage — a close-up, redundant camera |
| **Key finding 3** | From k≥4 the model reaches ≥97 % of the achievable ceiling ⇒ remaining error is not a fusion-coverage problem |
| **Also observed** | **Train** split reaches 100 % oracle coverage with just C1+C2+C3 (8566 instances); **test** needs all 7. Train/test coverage distributions differ |
| **Code verification** | Confirmed the ablation blanks views to **black images** (`camuvid.py:243-248`, `custom_datasets_fn.py:132-137`) and **still scores blanked cameras' GT** (`camuvid.py:328-330`, skip commented out). ⇒ Table 3 measures **missing-view robustness**, not **view-count scaling** |
| **Conclusion** | Strongest published evidence in this paper set that (a) coverage and disambiguation are separable contributions and (b) redundant views can actively degrade performance |
| **Next** | E3 |

---

## E3 — Cross-view scale variation (empirical test of MSMVD's premise)

| Field | Value |
|---|---|
| **Experiment ID** | E3 |
| **Paper** | P2 (premise under test); informs P1, P4 |
| **Date** | 2026-08-29 |
| **Code version** | `research_analysis/09_scripts/scale_variation.py` |
| **Dataset** | Wildtrack + MultiviewX shipped COCO GT (test) |
| **Method** | For every (frame, person) seen by ≥2 cameras, compute box-height ratio max/min across those cameras, and the number of distinct FPN levels the boxes would be assigned to (`level = clip(floor(4 + log2(√area/224)), 3, 7)`) |
| **Command** | `python research_analysis/09_scripts/scale_variation.py` |
| **Runtime** | ~5 s |
| **Output** | `research_analysis/04_results/E3_cross_view_scale_variation.txt` |
| **Result — Wildtrack** | 981 multi-view instances. Height ratio: median **2.73×**, mean 3.12×, p90 4.95×, p95 5.59×, max 7.72×. **90.3 %** have ≥2× ratio; 46.5 % ≥3×. **56.5 % would be assigned to ≥2 different FPN levels across views.** Per-camera median height 181 px (C6) → 344 px (C5) — 1.9× between cameras |
| **Result — MultiviewX** | 1494 instances. Median **2.89×**, p90 6.13×, max 8.34×. 73.0 % ≥2×. **30.1 %** span multiple FPN levels |
| **Expected result** | MSMVD asserts pedestrians have "vastly different scales between views" but provides no quantification |
| **Conclusion** | ✅ **MSMVD's premise is confirmed and is quantitatively large.** A single-scale projection (MVDet) is structurally mismatched to a regime where over half of all objects change pyramid level between cameras |
| **Caveat** | ⚠ The premise is *stronger* on Wildtrack (56.5 %) than MultiviewX (30.1 %), yet MSMVD's reported gain is *larger* on MultiviewX (+1.5 MODA) than Wildtrack (+0.5). Scale is therefore not the only driver of its improvement — an inconsistency the paper does not address |
| **Next** | E4 |

---

## E4 — Mining the ImGeoNet training logs

| Field | Value |
|---|---|
| **Experiment ID** | E4 |
| **Paper** | P3 |
| **Date** | 2026-08-29 |
| **Code version** | `research_analysis/09_scripts/log_class_stats.py` + PowerShell regex extraction |
| **Data** | `P3/ImGeoNet/logs/{scannet,arkit,scannet200}.txt` (932 KB of genuine mmdet training logs) |
| **Command** | `python research_analysis/09_scripts/log_class_stats.py` |
| **Runtime** | ~3 s |
| **Output** | `research_analysis/05_failure_analysis/E4_imgeonet_per_class_failures.txt` |
| **Recovered hardware** | ScanNet: **4× Tesla P40**, 2023-02-01, 10 776 MB, 1.76 s/iter, 901 iters/ep, ~5 h 52 m total. ARKit: **1× RTX 3090**, 2024-08-25, 19 417 MB, 2.66 s/iter, 1685 iters/ep, `samples_per_gpu=8`. ScanNet200: **3× RTX 3090**, 15 236 MB, 1.39 s/iter, 3603 iters/ep. All: PyTorch 1.7.1+cu110, MMCV 1.2.7, MMDet 2.10.0, MMDet3D 0.8.0 |
| **Result — paper vs log** | ScanNet mAP@.25: paper **54.8**, log **54.57** (ep 12) — README matches log exactly. mAP@.50: paper **28.4**, log **28.94**; note 28.40 is exactly the **epoch-9** value. ARKit: paper 60.2/43.4, log **59.82/42.76**. ScanNet200: paper 22.3, log **22.38** |
| **⚠ Finding 1** | **ScanNet200 training was abandoned at epoch 11 of 30.** Log ends with `Saving checkpoint at 11 epochs`, then the final eval table, with `eta: 1 day, 4:45:03` remaining. `lr_config step=[8,29]` — the second LR drop **never happened**. The published 22.38 / 9.67 comes from this ~37 %-complete run |
| **⚠ Finding 2** | `scannet.txt` declares `model.type='ImVoxelNet'` yet logs `loss_occ`/`acc_occ`; git hash `a75dd1a` vs `a5dfdf3` for the other two, 18 months apart. ⇒ The ScanNet log was produced by a **different code revision** than the released one (pre-rename) |
| **⚠ Finding 3** | Late-training volatility: ScanNet epochs 9–12 give 54.52 / 54.69 / 54.46 / 54.57 → **run noise ≈ ±0.2 mAP**. Differences below that are meaningless |
| **Result — failure modes** | **ScanNet (18 cls):** worst = picture **0.0427**, window 0.2466, curtain 0.3372, door 0.4041, counter 0.4117; best = toilet 0.9518, bed 0.8407, bathtub 0.8147. Mean AP@.25→AP@.50 drop **47.0 %**; worst localisation collapse showercurtain **−95.8 %**, curtain −87.4 %, counter −80.9 %. **ARKit (17 cls):** tv_monitor **0.0355 / AP@.50 = 0.0000**; stove 0.2360; dishwasher AR .88 but AP .20. **ScanNet200 (189 cls):** **63 classes (33.3 %) have AP@.25 = 0 and AR@.25 = 0** (never recalled once); 89 (47.1 %) have AP@.50 = 0; median class AP@.25 = **0.10** vs mean 0.2238; degenerate classes `scale` and `guitar` score AP@.25 = 1.0000 with AP@.50 = 0.0000 |
| **Conclusion** | Consistent, architecturally-explained failure mode: **thin / flat / wall-parallel objects fail** because they are thinner than one 0.16 m voxel and lie flush with a surface, so the geometry-shaping probability cannot separate them from the wall, and nearest-neighbour back-projection (`.round().long()`, `imgeonet.py:293-294`) quantises their extent. Recall survives, localisation collapses |
| **Assessment of the release** | Honest: README numbers match the logs to 2 decimals; the authors published actual runs, not cherry-picked bests. Paper numbers run 0.2–0.6 higher than the shipped logs, consistent with the README's own *"Performance may vary slightly depending on the number of GPUs"* |
| **Next** | E5 (requires GPU access) |

---

## Planned experiments — blocked pending GPU access

| ID | Paper | Description | Blocker | Priority |
|---|---|---|---|---|
| E5 | P5 | Reproduce CVT vehicle segmentation, Setting 2 (target 36.0 IoU) | 60 GB nuScenes + 361 MB labels; **no checkpoint exists → must train ~32 GPU-h** | **1st** — lightest software stack |
| E6 | P5 | View-count sweep via `data.cameras=[[1]] / [[1,4]] / [[0,1,2]] / [[0,1,2,3,4,5]]` | after E5 | **1st** — zero code change |
| E7 | P5 | View-order control: `data.cameras=[[5,3,1,0,4,2]]` — must be bit-identical (softmax over a set) | after E5 | high (cheap positive control) |
| E8 | P5 | Extract and analyse attention mass per (BEV cell, view); test whether it correlates with geometric visibility | after E5 | high — novel, data already in the forward pass |
| E9 | P3 | Reproduce ImGeoNet on **ARKitScenes** (single GPU, 12 ep, ~13 h, 19.4 GB — proven feasible by the log) | torch 1.7.1 + mmcv 1.2.7 + `nvcc` + ARKit download | 2nd |
| E10 | P3 | View-count sweep 10→100 (`n_images` in test pipeline only, no retraining) | after E9 | 2nd — reproduces paper Table 4 |
| E11 | P3 | **Feed `var_vols` to the detector** (`extract_feat(cat(avg*occ, var))`) vs matched baseline | after E9 | ⭐ **cheapest high-value experiment in the project** |
| E12 | P3 | Finish the abandoned ScanNet200 run to 30 epochs | 3 GPUs × ~1.5 days | low value, easy contribution |
| E13 | P1 | Reproduce MVDet Wildtrack 88.2 MODA | **2 GPUs** + MATLAB + Wildtrack | 3rd |
| E14 | P1 | View-order sensitivity (permute the camera loop) — should degrade, since fusion is order-dependent | after E13 | high — untested by any paper |
| E15 | P6 | Replace `.sum(1)` with `sum/valid_count` and with attention; measure on nuScenes | 8 GPUs + lidarseg + masks | low (resource-prohibitive) |
| E16 | — | Re-evaluate CaMuViD under the standard BEV 0.5 m protocol | CaMuViD weights + Wildtrack images | medium — needed for fair comparison |

---

## Log conventions

- Every entry records: ID · paper · date · code version · dataset · configuration · command · hardware · runtime · result · expected result · difference · observations · conclusion · next.
- **Never overwrite.** Corrections are appended as new entries referencing the old ID.
- Distinguish **[PAPER]** / **[CODE]** / **[LOG]** / **[OBS]** / **[INF]** in every claim.
- If an experiment cannot be run, log it as a failure with the exact blocker — do not leave it out.
