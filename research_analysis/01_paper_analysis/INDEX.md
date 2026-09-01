# Paper Analyses — Index (Step 3)

The full per-paper analyses live **inside each paper's own folder**, next to its PDF and code, so each folder is self-contained. Each report covers Steps 2, 3, 4, 5, 6, 9, 10, 11, 12, 13 and 16 for that paper.

| ID | Report | Paper | Venue | Relevance | Recommended role |
|---|---|---|---|---|---|
| P1 | [`../../P1/REPORT_P1_MVDet.md`](../../P1/REPORT_P1_MVDet.md) | MVDet — Multiview Detection with Feature Perspective Transformation | ECCV 2020 | 3.5 / 5 | Secondary baseline (multi-camera branch) |
| P2 | [`../../P2/REPORT_P2_MSMVD.md`](../../P2/REPORT_P2_MSMVD.md) | MSMVD — Multi-scale Image Features via Multi-scale BEV Features | arXiv 2025 (BMVC fmt) | 3 / 5 concept · 1 / 5 practical | **Idea source only — no code exists** |
| P3 | [`../../P3/REPORT_P3_ImGeoNet.md`](../../P3/REPORT_P3_ImGeoNet.md) | ImGeoNet — Image-induced Geometry-aware Voxel Representation | ICCV 2023 | ⭐ 4 / 5 | ⭐ **Primary baseline** |
| P4 | [`../../P4/REPORT_P4_CaMuViD.md`](../../P4/REPORT_P4_CaMuViD.md) | CaMuViD — Calibration-Free Multi-View Detection | CVPR 2025 | 2.5 / 5 | Critique / gap-evidence source |
| P5 | [`../../P5/REPORT_P5_CrossViewTransformers.md`](../../P5/REPORT_P5_CrossViewTransformers.md) | CVT — Cross-view Transformers | CVPR 2022 | ⭐ 4 / 5 (5 / 5 for its mechanism) | ⭐ **Architectural reference** |
| P6 | [`../../P6/REPORT_P6_GeoBEV.md`](../../P6/REPORT_P6_GeoBEV.md) | GeoBEV — Learning Geometric BEV Representation | AAAI 2025 | 2.5 / 5 | Supporting (borrow In-Box Label) |

## Structure of each report

1. Inventory
2. Relevance to *Multi-View Object Recognition* (A–E, with a relevance score under two readings of the topic)
3. Research problem (input / output / assumptions / constraints / objective)
4. Motivation and the exact research gap addressed
5. Previous approaches, grouped by category
6. Proposed method (architecture diagram + module-by-module table with file:line references)
7. **Multi-view fusion strategy** — mechanism, why it helps, weaknesses
8. Loss function (formulas, weights, verified against code)
9. Datasets
10. Experimental setup — **paper vs code, with mismatches flagged**
11. Attempt to execute — problem / cause / attempt / result table
12. Results reported vs reproduced
13. Controlled experiments the repo supports
14. Ablations available in the repository
15. Performance / cost — *is it the strategy or just size?*
16. Failure analysis
17. Limitations (methodological / dataset / computational / generalisation / implementation)
18. Research gaps — specific, with proposed directions
19. What to borrow
20. Verdict + cross-references to the other five papers

## Cross-cutting documents

- [`../06_comparison/master_comparison.md`](../06_comparison/master_comparison.md) — Step 14 (7 comparison tables)
- [`../02_code_analysis/code_trace_tables.md`](../02_code_analysis/code_trace_tables.md) — Step 4 (execution traces, file:line)
- [`../05_failure_analysis/failure_synthesis.md`](../05_failure_analysis/failure_synthesis.md) — Step 11
- [`../07_research_gaps/research_gaps.md`](../07_research_gaps/research_gaps.md) — Steps 12, 13, 15
- [`../08_research_directions/research_directions_and_roadmap.md`](../08_research_directions/research_directions_and_roadmap.md) — Steps 16, 17, 18
- [`../03_experiment_logs/experiment_log.md`](../03_experiment_logs/experiment_log.md) — Step 19
- [`../00_master_summary.md`](../00_master_summary.md) — Steps 1, 20
