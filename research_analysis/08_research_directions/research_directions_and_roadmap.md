# Research Directions and Roadmap (Steps 16, 17, 18)

> ⚠ **Novelty caveat, stated once and applying throughout.** Everything below is checked against **these six papers and the related work they cite**. That is *not* a literature review. Before claiming novelty for any idea here you must search at minimum: MVCNN / GVCNN / view-GCN / MVTN (multi-view 3D shape recognition), the *active view selection* and *next-best-view* literature, set-transformer / deep-sets aggregation, and recent BEV survey papers (2023–2026). **Items are marked 🔍 where I consider a prior-art collision most likely.**

---

## Part 1 — What to borrow from each paper (Step 16)

| Paper | Useful idea | Why it works (evidence) | Limitation as published | How it could inspire your research |
|---|---|---|---|---|
| **P1 MVDet** | Project **feature maps**, not pixels or detections | **[PAPER]** 88.2 vs 68.2 vs 26.8 MODA with the *same* network — a clean controlled attribution | Single scale; single ground plane; zero-pads outside FoV | Keep the principle; project **multiple scales** and carry an explicit **validity mask** so "outside FoV" ≠ "empty" |
| **P1** | Large-kernel dilated convs replace CRF | **+11.3 MODA**; end-to-end, no hand-designed potentials | ~19 M params at N=7 | Keep a wide post-fusion receptive field; consider separable/dilated or a small BEV transformer for the same RF at lower cost |
| **P2 MSMVD** | **Multi-Scale Projection** — BEV features inherit their source image scale | **+3.1 MODA** alone; **[OBS]** justified: **56.5 %** of Wildtrack pedestrians span ≥2 FPN levels across views | No code; cost never measured; 15 warps/view/frame | ⭐ Highest-value idea available. Its absence of an implementation makes a careful re-implementation legitimate work |
| **P2** | **BEV-FPN** — mix scales *after* view aggregation | **+5.3 MODA** | — | Directly transferable to any BEV/voxel representation |
| **P2** | Parameter-matched backbone sweep (R18→R101) | MSMVD-R18 (22.9 M, 80.2) beats baseline-R101 (43.2 M, 75.8) | — | ⭐ **Copy this protocol.** It is the standard answer to "isn't it just bigger?" |
| **P3 ImGeoNet** | **Masked mean, normalised by `valid_count`** | ⭐ Enables **train-20 / test-50 views** — impossible for P1/P4 | Uniform weighting | ⭐ **Make this your default fusion**, then add learned weights on top |
| **P3** | **Cross-view variance** as a free geometry cue | Geometry shaping worth **+6.1/+4.6 mAP** for **+26 ms constant** (vs 15.7× for an MVS-depth cascade) | **[CODE]** variance reaches only the occupancy head, never the detector | ⭐ Feed it to the recogniser too — see Idea 1 |
| **P3** | Depth as **training-only** supervision | Image-only at inference, verified in code (ARKit test pipeline collects `keys=['img']`) | Still needs RGB-D to train | Excellent pattern: supervise with a modality you will not have at test time |
| **P3** | Shipping **matched baseline configs** | Ablations become config diffs | — | ⭐ Copy this discipline |
| **P4 CaMuViD** | **Calibration-free** objective | Cross-dataset MODA 86.4 vs MVFP 76.7 **[PAPER]** | **[CODE]** achieved by removing alignment altogether (channel-mixing, no spatial warp) | Keep the goal, change the mechanism → learn correspondence by attention |
| **P4** | Concat > sum, with a stated reason | *"a high-activation area summed to a low-activation area would force the two activations to the middle"* — **+0.8 MODA** | Order-dependent, O(N) params | **[INF] Their argument is really an argument for *weighting*, not concatenation.** A confidence-weighted sum gets the benefit and stays permutation-invariant — a clean, testable hypothesis |
| **P4** | Shipping the **annotation JSONs** | Enabled 3 original analyses here with **no GPU, no images, no model** | — | ⭐ Copy this practice |
| **P5 CVT** | ⭐ **Softmax attention over the union of (views × patches)** | Simultaneously permutation-invariant, N-agnostic, **and a learned per-cell view selector**. 5 M params, 35 FPS, matches a 7 M / 8 FPS competitor with the *same* backbone and decoder | Quadratic in BEV grid size (hence 25×25 queries); normalises evidence count away | ⭐ **Make this your fusion backbone**, then fix the count problem |
| **P5** | **Calibration as a positional embedding**, not a warp | **+5.0 IoU** over appearance-only keys | Assumes near-static calibration | Generalises to *any* multi-view setting, including object-centric capture where "camera pose" = viewpoint |
| **P5** | Learned per-camera embedding recovers **34.4 of 36.0 IoU** | Table 4 | Not followed up by anyone | 🔍 A **semi-calibration-free** path: learn per-camera codes, fine-tune on a new rig with few labels |
| **P5** | Reporting params + FPS + GPU-hours beside accuracy | Turns a 2nd-place IoU (37.5 vs 37.7) into a 1st-place contribution | — | ⭐ Copy this reporting protocol |
| **P6 GeoBEV** | ⭐ **In-Box Label** — supervise the object's *interior*, not its visible surface | **+2.2 mAP / +2.2 NDS with zero added parameters** | Needs GT boxes + instance masks; helps size more than orientation | ⭐ Directly applicable to **[LOG]** ImGeoNet's thin-object failure (picture AP **0.043**) — a surface-vs-volume problem |
| **P6** | "**Gather, don't scatter**" (RC-Sampling) | Bilinear sampling from a dense radial map cannot leave holes; fastest *and* most accurate in its Table 4 | Polar→Cartesian interpolation blurs the far field | Good default rule for any lift-to-3D pipeline |
| **P6** | Plug-in validation across **3 independent baselines** | BEVDepth +4.4, BEVDet +2.7, BEVStereo +3.4 mAP | — | ⭐ **Copy this protocol.** Far more convincing than one tuned number |

---

## Part 2 — Candidate research directions, ranked (Step 17)

### Ranking table

| # | Idea | Novelty | Feasibility (no/1 GPU) | Expected impact | Compute cost | Data needs | Difficulty | **Priority** |
|---|---|---|---|---|---|---|---|---|
| **1** | **Set-statistics fusion**: replace mean/max/sum with (mean, var, max, valid-count) | Medium 🔍 | ⭐ **High** | Medium-High | ⭐ Very low | reuse | ⭐ Low | ⭐⭐⭐⭐⭐ **1st** |
| **2** | **Monotonic view aggregation**: guarantee adding a view never hurts | ⭐ High 🔍 | High | ⭐ High | Low | reuse | Medium | ⭐⭐⭐⭐⭐ **2nd** |
| **3** | **Complementarity-based view selection**: match 50-view accuracy with ~25 views | Medium 🔍 | High | High (efficiency) | Low | reuse | Medium | ⭐⭐⭐⭐ **3rd** |
| **4** | **Rig-redundancy-normalised benchmarking protocol** | Medium | ⭐⭐ **Very high (no GPU)** | Medium (community) | ⭐ None | annotations only | ⭐ Very low | ⭐⭐⭐⭐ **4th** |
| **5** | **Scale-aware attention fusion** (P2 × P5) | Medium-High 🔍 | Medium | High | Medium-High | reuse | High | ⭐⭐⭐ 5th |
| **6** | **Interior supervision for thin objects** (P6's In-Box → P3's occupancy) | Medium 🔍 | Medium | Medium-High | Low | ScanNet | Medium | ⭐⭐⭐ 6th |
| **7** | **Attention-mass analysis**: does CVT learn rig geometry? | Medium | High (after E5) | Medium (insight) | ⭐ Very low | reuse | Low | ⭐⭐⭐ 7th |
| **8** | **Semi-calibration-free per-camera embeddings** | Medium 🔍 | Medium | Medium | Medium | reuse | Medium-High | ⭐⭐ 8th |
| **9** | **Unified evaluation harness** (BEV-distance + image-IoU) | Low | High | Medium | Low | needs weights | Medium | ⭐⭐ 9th |
| **10** | **Normalise GeoBEV's `.sum(1)`** on a high-overlap rig | Low-Medium | ⭐ Low (8 GPUs) | Medium | Very high | nuScenes+ | Medium | ⭐ 10th |

---

### ⭐ Idea 1 — Set-statistics fusion *(start here)*

**Gap addressed:** Gaps 1 and 3 — cross-view agreement is never available to the recogniser.

**The observation.** Every method reduces the view set with a *single* statistic: mean (P3), max (P2), sum (P6), or concat (P1, P4). All of them are lossy in the same way — after reduction you cannot tell *one strong view* from *six agreeing views*. **[CODE]** ImGeoNet already computes the variance (`imgeonet.py:110`) — and then at `L142/L167` forwards only `avg_vols * occ` to the detector. The signal exists and is thrown away.

**Proposal.** Replace the single reduction with a concatenated **set-statistics vector** per location:
```
z = concat( mean_t(V_t),  var_t(V_t),  max_t(V_t),  valid_count/N )
```
Permutation-invariant, N-agnostic, ~4× the fused channel width, no new hyper-parameters.

**Why it should work.** **[OBS]** E2 shows cameras C4/C5 add **+4.8 pp recall at zero coverage gain** — pure corroboration. A statistic that encodes *how many views agree* is the natural representation for that.

**Cost.** A ~5-line change in `imgeonet.py` plus a widened `neck_3d` input. ImGeoNet ships a matched `imvoxelnet_*.py` baseline to compare against.

**Risks.** Variance is scale-dependent and may need normalisation; a 4× wider fusion input adds parameters (must be controlled with a width-matched baseline, per P2's protocol). 🔍 Deep-Sets / set-transformer literature has proposed multi-statistic pooling generally — **your contribution would be the multi-view-recognition-specific instantiation and evaluation, not the pooling idea itself.** Frame it that way.

---

### ⭐ Idea 2 — Monotonic view aggregation *(the most defensible contribution)*

**Gap addressed:** Gap 2 — more views can hurt.

**The observation.** **[PAPER]** P4 Table 3: 6 cameras → **95.6 MODA**; 7 cameras → **95.0**. Adding a camera *reduced* accuracy. **[OBS]** E2 confirms C7 adds only +0.40 pp of oracle coverage while recall falls 0.7 pp. **[OBS]** E1 shows C7 is a close-up camera (largest mean box area, 40 595 px) with only 62.2 % coverage.

**Proposal.** A fusion operator with a **gate** `g_t ∈ [0,1]` per (location, view), predicted from that view's projected feature, its validity mask, and its geometric footprint:
```
z = Σ_t g_t · V_t  /  Σ_t g_t          (with Σ g_t ≥ ε)
```
Train it with an explicit **monotonicity objective**: sample a random view subset `S ⊂ S'` per iteration and penalise `max(0, loss(S') − loss(S))` — i.e. adding views must not increase loss.

**Pre-registered success criterion (state this before running):**
> On Wildtrack, **7-camera performance ≥ 6-camera performance**, while matching or beating the 6-camera baseline of 95.6 MODA.

That is a crisp, falsifiable claim that no existing method satisfies, and the counter-example is *already published by the paper you would be improving on*.

**Why this is your strongest option.** It attacks a documented failure with an explicit, measurable target; the baseline number to beat is public; and the mechanism (gating + subset-consistency training) is simple enough to implement in weeks.

🔍 **Check first:** monotonic/submodular set functions, and any "view dropout consistency" training in multi-view shape recognition (MVTN and view-GCN both touch view weighting).

---

### ⭐ Idea 3 — Complementarity-based view selection

**Gap addressed:** Gap 2 — redundant views cost linear compute for no gain.

**The observation.** **[PAPER]** P3 Table 4: ARKit 10→20 views = **+14.1** mAP; 50→100 = **+2.2**. **[OBS]** E1: three Wildtrack cameras cover 99.5 % of pedestrians. Yet ImGeoNet samples views by plain `np.linspace` (**[CODE]** `multi_view.py:25`) — uniform, content-blind.

**Proposal.** Score candidate views for **complementarity** (pose diversity + predicted marginal coverage gain, computed from the running fused representation), select greedily, and stop when marginal gain falls below a threshold.

**Pre-registered success criterion:**
> Match ImGeoNet's 50-view accuracy (**60.2 / 43.4** mAP on ARKit) using **≤ 25 selected views**, i.e. ~2× inference speed-up at equal accuracy.

The baseline curve is published (P3 Table 4), so the comparison is free. **[CODE]** `MultiViewPipeline` makes the selector a ~5-line swap, and ImGeoNet's N-agnostic mean pooling means no retraining is needed to *evaluate* a selector.

🔍 **Check first:** this overlaps the **next-best-view / active vision** literature substantially. Your differentiator must be that selection is *learned jointly with recognition* and evaluated on a *published saturating curve*, not that view selection is new.

---

### ⭐ Idea 4 — Rig-redundancy-normalised benchmarking *(do this first — it needs no GPU)*

**Gap addressed:** Gap 5.

**The observation.** **[INF]** The dropout curve is **linear** for nuScenes (P5 Fig. 4) and **non-monotonic** for Wildtrack (P4 Table 3) — because the rigs differ in redundancy, not because the methods differ. No paper reports a redundancy statistic.

**Proposal.** Define and publish two annotation-only statistics per dataset:
- **mean views per object** (Wildtrack **5.26/7**, MultiviewX **4.97/6** — already computed)
- **pairwise view Jaccard matrix** (Wildtrack C1↔C6 **92.8 %** — already computed)

plus the **oracle coverage curve** vs. camera subset. Report all view-count experiments against them.

**Why this is worth doing now.** It costs nothing (`09_scripts/view_coverage.py` already works), it makes every subsequent view-count claim you make falsifiable, and it protects your own results from the criticism you are levelling at others. It also makes a good short workshop paper on its own.

---

### Idea 5 — Scale-aware attention fusion (P2 × P5)

Combine MSMVD's per-scale projection with CVT's cross-view attention: attend over the **(view × scale × patch)** set instead of max-pooling views then FPN-ing scales.

**Motivation:** **[OBS]** E3 — 56.5 % of Wildtrack pedestrians span multiple FPN levels across views, so the *right scale* is view-dependent. Attention can select scale and view jointly; MSMVD's max-then-FPN cannot.

**Risk:** highest engineering cost here (requires re-implementing MSP from text alone, §13.4 of the P2 report). **[INF] Do Ideas 1–2 first; treat this as the follow-up paper.**

---

### Idea 6 — Interior supervision for thin structures

Transplant **[PAPER]** P6's In-Box Label (+2.2 mAP, **zero parameters**) into P3's occupancy head, replacing binary surface occupancy with a signed-distance or interior-filling target, and replacing `.round().long()` back-projection (**[CODE]** `imgeonet.py:293`) with trilinear sampling.

**Target:** **[LOG]** ScanNet `picture` AP@0.25 = **0.0427**, `showercurtain` AP@0.25 0.475 → AP@0.50 0.020 (**−95.8 %**). Both are surface-vs-volume and quantisation failures. Also relevant: **[PAPER]** P3's oracle-depth ceiling is **4.0 / 5.0 mAP above** what geometry shaping achieves — headroom the paper itself flags.

---

### Idea 7 — Attention-mass analysis (cheap insight, do alongside E5)

**[CODE]** CVT computes `att` at `encoder.py:158` and discards it. Aggregate it into a per-(BEV-cell, view) attention mass and ask: does it recover the rig geometry? Does it down-weight occluded views? Can it be **supervised** with the nuScenes visibility labels CVT already loads for `min_visibility`?

Low cost, produces figures, and directly supports Ideas 1–3 by showing whether "learned view weighting" actually learns visibility.

---

## Part 3 — Recommended baselines

| Role | Paper | Why |
|---|---|---|
| ⭐ **Primary baseline** | **P3 ImGeoNet** | Only repo that is genuinely N-agnostic and permutation-invariant (**[CODE]** train-20/test-50), ships **matched ablation configs**, ships **three real training logs**, publishes the **views-vs-accuracy curve** you need, and predicts **object categories**. Its weakness (uniform mean; variance discarded) *is* your topic. ⚠ No checkpoints released → you must train. **Start with ARKitScenes**: **[LOG]** proven to complete on **one RTX 3090**, 12 epochs, ~13 h, 19.4 GB |
| ⭐ **Architectural reference / secondary baseline** | **P5 CVT** | The only learned fusion mechanism in the set; lightest software stack (pure pip, **no custom CUDA ops**, single GPU, `overfit.py`); `cameras` config makes view-count/selection/order experiments free. ⚠ No checkpoints; 60 GB nuScenes; fix `torchmetrics.compute_on_step` |
| **Secondary baseline (multi-camera branch)** | **P1 MVDet** | The canonical baseline everyone compares to; complete, faithful code; `--variant` ablations all runnable. ⚠ Needs **2 GPUs** and MATLAB for the official metric |
| **Supporting — idea source only** | **P2 MSMVD** | Best idea (multi-scale projection), **no code**. Use as motivation and as a re-implementation target, not a baseline |
| **Supporting — critique / evidence source** | **P4 CaMuViD** | Its camera-elimination table plus my oracle cross-check (E2) is your **strongest evidence** for the research gap. ⚠ Do not use as a numeric baseline — protocol not comparable (**[CODE]** image-space IoU vs BEV distance) |
| **Less relevant** | **P6 GeoBEV** | Excellent engineering, but **[CODE]** its cross-view fusion is a bare `.sum(1)` — the contribution is lifting + supervision, not fusion. Heaviest setup (8 GPUs, 3 data modalities, custom CUDA op). Borrow **In-Box Label** and the plug-in validation protocol; skip the rest |

---

## Part 4 — Research roadmap (Step 18)

```
   [DONE] Paper analysis (6/6)  +  Code trace (5/5)  +  Gap identification
                                   │
                                   ▼
   [DONE, no GPU needed] Data-only evidence:  E1 view coverage · E2 oracle vs ablation · E3 scale variation · E4 log mining
                                   │
                                   ▼
   ┌──────────────── PHASE 0 (weeks 1–2, NO GPU) ────────────────┐
   │ • Idea 4: rig-redundancy protocol on Wildtrack/MVX/nuScenes │
   │ • Secure GPU access; decide branch (see Decision below)      │
   │ • Download ARKitScenes (P3) or nuScenes (P5)                 │
   └───────────────────────────┬─────────────────────────────────┘
                               ▼
   ┌──────────────── PHASE 1 (weeks 3–6) BASELINE REPRODUCTION ──┐
   │ E9  ImGeoNet on ARKit  → target 59.8 / 42.8 (log-verified)  │
   │ E5  CVT on nuScenes    → target 36.0 IoU  (~32 GPU-h)       │
   │ ⚠ Accept ±0.2–0.6 as run noise ([LOG] E4 measured ±0.2)     │
   └───────────────────────────┬─────────────────────────────────┘
                               ▼
   ┌──────────────── PHASE 2 (weeks 7–10) CONTROLLED EXPERIMENTS ┐
   │ E10 view-count sweep 10→100 (P3, no retraining)             │
   │ E6  view-count via data.cameras (P5, no code change)        │
   │ E7  view-ORDER control — must be bit-identical              │
   │ E8  attention-mass analysis (Idea 7)                        │
   │ Missing-view study in all three regimes (blank / mask /     │
   │ retrain) — the distinction Gap 4 says nobody makes          │
   └───────────────────────────┬─────────────────────────────────┘
                               ▼
   ┌──────────────── PHASE 3 (weeks 11–14) FIRST CONTRIBUTION ───┐
   │ E11 Idea 1: set-statistics fusion vs matched baseline       │
   │     (⭐ cheapest high-value experiment in the project)       │
   │ Ablate each statistic; width-matched control (P2 protocol)  │
   └───────────────────────────┬─────────────────────────────────┘
                               ▼
   ┌──────────────── PHASE 4 (weeks 15–22) MAIN CONTRIBUTION ────┐
   │ Idea 2: monotonic gated aggregation                          │
   │ Pre-registered target: 7-cam ≥ 6-cam, and ≥ 95.6 MODA        │
   │ + Idea 3 if time: selection at ≤25 views ≈ 50-view accuracy  │
   └───────────────────────────┬─────────────────────────────────┘
                               ▼
   ┌──────────────── PHASE 5 (weeks 23–28) VALIDATION ───────────┐
   │ Plug-in validation on ≥2 baselines (P6's protocol)          │
   │ Full ablation · params/FLOPs/FPS table (P5's protocol)      │
   │ Failure analysis · cross-dataset · write-up                 │
   └─────────────────────────────────────────────────────────────┘
```

### ⚠ Decision you must make before Phase 1

**Which reading of "Multi-View Object Recognition" is yours?**

| If you mean… | Primary baseline | Dataset | Which ideas apply |
|---|---|---|---|
| **(a) Many viewpoints of the same objects → categories** (closest to MVCNN-style recognition) | **P3 ImGeoNet** | ARKitScenes → ScanNet | 1, 2, 3, 6, 7 |
| **(b) Multiple fixed cameras of a scene → detect/recognise objects** | **P1 MVDet** (+ P5's fusion) | Wildtrack, MultiviewX, **GMVD** | 1, 2, 4, 5 |
| **(c) Both — a fusion operator that works in either** | **P5 CVT** encoder + P3 task | ARKit + Wildtrack | 1, 2, 4 |

**My recommendation: (a) or (c).** Reading (a) gives you real category recognition, variable N, and a published views-vs-accuracy curve. Reading (c) is the strongest *contribution* framing — a fusion operator validated across two rig types would directly demonstrate the redundancy law in Gap 5 — but costs two environments.

---

## Part 5 — Expected contributions *(potential — none validated yet)*

Label these as **potential** in your writing until experiments confirm them.

1. **A quantitative characterisation of view redundancy in standard multi-view benchmarks.** *(Partly complete — E1/E2/E3 already computed.)* Wildtrack: 3 cameras cover 99.5 % of pedestrians; C2/C4/C5 add 0.00 pp; the 7th camera reduces published MODA. MultiviewX: 2 cameras cover 100 %. Median cross-view scale ratio 2.7–2.9×.
2. **The first explicit separation of *coverage* from *disambiguation* in multi-view detection**, with evidence that views 4–5 on Wildtrack contribute +4.8 pp recall at **zero** coverage gain. 🔍 verify against the multi-view detection literature.
3. **A set-statistics multi-view aggregation operator** that preserves cross-view agreement, with an ablation isolating mean / variance / max / count.
4. **A monotonic view-aggregation operator** with a pre-registered guarantee that adding a view never degrades accuracy — tested against a published counter-example.
5. **A complementarity-based view selector** achieving ~2× inference speed-up at equal accuracy on a published saturating curve.
6. **A rig-redundancy-normalised evaluation protocol**, making "our fusion scales better with views" falsifiable for the first time.
7. **Reproducibility findings** *(already established this session, publishable as an appendix)*: ImGeoNet's ScanNet200 number comes from an **epoch-11-of-30** run whose second LR drop never fired **[LOG]**; MVDet's evaluator **silently** falls back from MATLAB to a Python re-implementation the repo itself says is 0–2 MODA lower **[CODE]**; CaMuViD's MODA is computed under an **image-space IoU** protocol unlike every method in its own comparison table **[CODE]**.

---

## Part 6 — Rules to hold yourself to

Carried from the six papers' best and worst practices:

1. **Change one variable at a time.** P1's projection ablation is the model: same network, same training, three projection choices.
2. **Always ship a width/parameter-matched baseline** (P2's Table 4).
3. **Validate as a plug-in on ≥2 independent baselines** (P6's Table 3).
4. **Report params, FLOPs, FPS and GPU-hours beside accuracy** (P5).
5. **Make ablations config diffs, not code branches** (P3, P5).
6. **State the evaluation protocol explicitly** and re-evaluate any baseline that used a different one (P1's silent fallback, P4's image-space IoU).
7. **Report run-to-run noise.** **[LOG]** ImGeoNet's last four ScanNet epochs span ±0.2 mAP — never claim a smaller difference.
8. **Report rig redundancy with every view-count curve** (Gap 5 — your own proposal; apply it to yourself).
9. **Distinguish blanked / masked / retrained** whenever you reduce the view count (Gap 4).
10. **Never report a number from an unconverged run without saying so** (the ScanNet200 lesson).
