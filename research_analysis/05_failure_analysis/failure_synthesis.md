# Cross-Paper Failure Analysis (Step 11)

Raw evidence: `E4_imgeonet_per_class_failures.txt` (this folder) and `../04_results/E1–E3`.

Tags: **[PAPER]** claimed · **[CODE]** verified in source · **[LOG]** shipped training logs · **[OBS]** measured by me · **[INF]** inference

---

## 1. The evidence base, and its limits

| Paper | Predictions shipped? | Logs shipped? | Failure evidence available |
|---|---|---|---|
| P1 MVDet | ✗ | ✗ | paper figures/tables only |
| P2 MSMVD | ✗ (no code) | ✗ | paper only |
| **P3 ImGeoNet** | ✗ | ⭐ **✅ 3 logs, per-class AP/AR** | ⭐ **quantitative, per class** |
| P4 CaMuViD | ✗ | ✗ | ⭐ **GT annotations → my E1/E2** |
| P5 CVT | ✗ | ✗ | paper Fig. 3/4 only |
| P6 GeoBEV | ✗ | ✗ | paper sub-metrics only |

**[INF]** Only P3 permits genuine failure analysis. Sections 2–3 are therefore heavily P3-weighted, and I say so rather than manufacturing balance.

---

## 2. Failure mode A — **thin, flat, surface-flush objects** ⭐ the dominant pattern

**[LOG]** ScanNetV2, final epoch (mAP@0.25 = 54.57):

| Class | AP@0.25 | AR@0.25 | AP@0.50 | Geometry |
|---|---|---|---|---|
| **picture** | **0.0427** | 0.1351 | 0.0025 | flat, on a wall |
| window | 0.2466 | 0.5213 | 0.0294 | flat, in a wall |
| curtain | 0.3372 | 0.6269 | 0.0426 | thin, against a wall |
| door | 0.4041 | 0.6510 | 0.0804 | flat, in a wall |
| counter | 0.4117 | 0.7308 | 0.0786 | flush with cabinetry |
| showercurtrain | 0.4746 | 0.6786 | **0.0199** | thin, against a wall |
| — | | | | |
| toilet | **0.9518** | 0.9655 | 0.6883 | bulky, free-standing |
| bed | 0.8407 | 0.8519 | 0.7476 | bulky, free-standing |
| bathtub | 0.8147 | 0.8387 | 0.6165 | bulky, free-standing |

**[LOG]** ARKitScenes reproduces it exactly: `tv_monitor` **AP@0.25 = 0.0355, AP@0.50 = 0.0000** (wall-mounted flat panel); `stove` 0.2360 (flush with counter); `dishwasher` **AR@0.25 = 0.88 but AP@0.25 = 0.20** — found reliably, but massively over-detected.

**[INF] Root cause is architectural, and traceable to two specific lines:**
1. **Voxel size 0.16 m** (config) exceeds the object's thickness. A picture occupies a sub-voxel slab flush with the wall, so the geometry-shaping surface probability `S` cannot separate "picture" from "wall", and the mean-pooled feature averages both.
2. **Nearest-neighbour back-projection** — **[CODE]** `imgeonet.py:293-294` uses `.round().long()`, so extents are quantised to the voxel grid.

**Cross-paper corroboration:** **[PAPER]** P6 identifies the same class of problem from the opposite direction — LiDAR depth labels record only the **ego-facing surface**, not the object's volume — and fixes it with the In-Box Label for **+2.2 mAP at zero parameters**. **[INF] P6's fix is directly applicable to P3's failure. Neither paper cites the other.**

---

## 3. Failure mode B — **localisation collapse from IoU 0.25 → 0.50**

**[LOG]** Mean AP drop across all 18 ScanNet classes: **47.0 %**. Worst cases:

| Class | AP@0.25 → AP@0.50 | Relative drop |
|---|---|---|
| showercurtrain | 0.4746 → 0.0199 | **−95.8 %** |
| curtain | 0.3372 → 0.0426 | −87.4 % |
| counter | 0.4117 → 0.0786 | −80.9 % |
| door | 0.4041 → 0.0804 | −80.1 % |
| cabinet | 0.4059 → 0.1578 | −61.1 % |

ARKit is milder (mean −28.5 %) but shows the same ordering: sink −66.3 %, shelf −61.7 %, fireplace −59.3 %.

**[INF]** Recall survives while precision-at-tight-IoU collapses ⇒ **the model finds these objects but cannot size them.** Consistent with mode A and with **[PAPER]** P3's own admission that the GT-depth oracle reaches **58.8 / 33.4** vs ImGeoNet's 54.8 / 28.4 — a **4.0 / 5.0** gap the paper flags as *"room for improving Geometry Shaping Network in the future"*.

---

## 4. Failure mode C — **long-tail collapse**

**[LOG]** ScanNet200, 189 evaluated classes:
- **63 classes (33.3 %)** have **AP@0.25 = 0.0000 and AR@0.25 = 0.0000** — never recalled once.
- **89 classes (47.1 %)** have AP@0.50 = 0.
- **Median** class AP@0.25 = **0.10** vs **mean** 0.2238 — heavily skewed.
- ⚠ Degenerate inflation: `scale` and `guitar` both score **AP@0.25 = 1.0000 with AP@0.50 = 0.0000** — near-certainly 1–2 test instances each.

**[INF]** The headline 22.38 mAP is carried by a minority of well-represented classes plus a handful of degenerate single-instance classes. **This is a resolution + class-imbalance failure, not a fusion failure** — important to state, because it would be easy to mis-attribute to the multi-view design.

---

## 5. Failure mode D — **redundant and harmful views**

**[PAPER]** P4 Table 3, cross-checked against **[OBS]** E2:

| added | Δ oracle coverage | Δ recall | Δ MODA |
|---|---|---|---|
| C4 | **0.00 pp** | +3.4 | +3.2 |
| C5 | **0.00 pp** | +1.4 | +0.0 |
| **C7** | +0.40 pp | **−0.7** | **−0.6** |

**[OBS]** E1 explains C7: largest mean box area (**40 595 px** — a close-up camera) and only 62.2 % coverage. **[INF]** It contributes duplicate detections of already-visible people; under a fixed-N concat architecture there is no mechanism to down-weight it.

**Saturation elsewhere:** **[PAPER]** P3 Table 4 — ARKit gains **+14.1** mAP for views 10→20 but only **+2.2** for 50→100.

---

## 6. Failure mode E — **cross-view scale mismatch**

**[OBS]** E3, from Wildtrack/MultiviewX ground truth:

| | Wildtrack | MultiviewX |
|---|---|---|
| Median cross-view box-height ratio (same person) | **2.73×** | 2.89× |
| ≥ 2× ratio | **90.3 %** | 73.0 % |
| Would land on **≥2 different FPN levels** across views | **56.5 %** | 30.1 % |

**[INF]** P1 (single dilated scale) and P4 (FPN levels fused **independently**, never interacting) are structurally unable to represent this. P2 is the only paper that addresses it — and has no code.

---

## 7. Failure mode F — **far-field degradation (surround rigs)**

**[PAPER]** P5 Fig. 3: IoU falls from ~36 at 0 m to **0 by ~60 m**; CVT is *below* FIERY beyond ~50 m. **[PAPER]** §5.4 explains: *"Partially occluded far-away samples have fewer corresponding image features... There is less training data and fewer geometric priors to rely upon."*

**[PAPER]** P6's weakest sub-metric is orientation: GeoBEV wins mAP, NDS, mATE and mASE but **loses mAOE** (0.318 vs RayDN's 0.315). **[INF]** In-Box Label fills the box *interior* isotropically (the CAI weight is `∛` of three symmetric ratios) — it improves **size** (best mASE 0.254) but carries **no directional signal**, so orientation is left under-constrained.

---

## 8. Consolidated failure table

| Failure type | Example (evidence) | Possible cause | Potential solution |
|---|---|---|---|
| **Thin / flat / surface-flush objects** | picture 0.043, tv_monitor 0.036 **[LOG]** | Object thinner than one 0.16 m voxel; surface probability cannot separate it from the wall; mean pooling averages object + wall | Finer/anisotropic voxels; **interior (In-Box) supervision** from P6; signed distance instead of binary occupancy |
| **Localisation collapse @ IoU 0.5** | showercurtain −95.8 %, mean −47.0 % **[LOG]** | Nearest-neighbour back-projection **[CODE]** `.round().long()`; voxel quantisation | Trilinear back-projection; box refinement head; higher-resolution volume near detections |
| **Long-tail never recalled** | 63/189 classes at AP=AR=0 **[LOG]** | Small rare objects; class imbalance; resolution | Not a fusion problem — resolution + sampling; report median as well as mean AP |
| **Redundant view hurts** | 7 cams 95.0 < 6 cams 95.6 MODA **[PAPER]**; C7 adds 0.40 pp coverage **[OBS]** | Duplicate evidence; uniform fusion cannot down-weight | ⭐ **Learned view gating with a monotonicity constraint** (Idea 2) |
| **Coverage/disambiguation conflated** | C4, C5 give +4.8 pp recall at **0.00 pp** coverage **[OBS]** | Fusion has no notion of corroboration | ⭐ **Set statistics: variance + valid-count** (Idea 1) |
| **Cross-view scale mismatch** | 56.5 % span ≥2 FPN levels **[OBS]** | Single-scale or independently-fused scales | **Multi-scale projection** (P2's MSP) |
| **Far-field / occluded objects** | IoU → 0 by 60 m **[PAPER]** | Few image features; weak priors | Distance-aware loss; temporal fusion; explicit far-field depth prior |
| **Orientation under-constrained** | mAOE 0.318, GeoBEV's only losing metric **[PAPER]** | CAI weight is isotropic | Anisotropic / directional inner weighting |
| **Out-of-FoV ≡ empty** | **[CODE]** `warp_perspective` zero-pads; no validity mask into `map_classifier` | No visibility input to fusion | **Explicit validity mask** (P3 does this with `valid_count`; P1, P6 do not) |
| **Unnormalised cross-view sum** | **[CODE]** `rcsample.py:378` `.sum(1)` | Overlap wedges get doubled magnitude, purely from rig geometry | Divide by valid-view count; matters more as rig overlap grows |
| **Run-to-run noise mistaken for gains** | ScanNet ep 9–12: 54.52 / 54.69 / 54.46 / 54.57 **[LOG]** | batch size 1, `RepeatDataset ×3` | ⭐ **Report ±0.2; never claim a smaller difference** |

---

## 9. What the failures collectively say

1. **Fusion is not where these models fail.** They fail at **geometric resolution** (thin objects, tight-IoU localisation) and at **class imbalance**. The exception is failure mode D — where **fusion is exactly the problem**, and it is the mode nobody has addressed.
2. **Recall is usually fine; precision at tight tolerance is not.** Across P3's three datasets, AR@0.25 comfortably exceeds AP@0.25 for the failing classes. The models *find* things and *mis-size* them.
3. **The one failure that is unambiguously a multi-view fusion failure — a redundant camera reducing accuracy — has a published example, a measurable target (95.6 MODA at 6 cameras), and no proposed solution.** That is where a contribution has the clearest evidence behind it.
