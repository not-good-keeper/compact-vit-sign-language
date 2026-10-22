# Final technical report: ISL-ViT-Tiny / ISL Reader

Evidence-audited report for 23CSE473 Neural Networks and Deep Learning (Amrita School of Computing, Coimbatore), written against commit `ea092f3e25124fcd8929929900e53879d7084943` of this repository (`main`, 23 October 2026).

| Deliverable | File |
|---|---|
| Report (PDF) | [`ISL_ViT_Tiny_Final_Report.pdf`](ISL_ViT_Tiny_Final_Report.pdf) (also `latex/main.pdf`) |
| LaTeX source, bibliography, template | `latex/main.tex`, `latex/references.bib`, `latex/sn-jnl.cls`, `latex/sn-mathphys-num.bst`, `latex/build.sh` |
| Figures (vector PDF and PNG preview) | `latex/figures/fig_*.pdf`; diagrams are TikZ inside `latex/main.tex` |
| Model comparison (machine-readable) | [`model_comparison.csv`](model_comparison.csv) |
| Evidence ledger | [`evidence_ledger.md`](evidence_ledger.md) |
| Discrepancy register | [`discrepancy_register.md`](discrepancy_register.md) |
| Rubric compliance matrix | [`rubric_compliance.md`](rubric_compliance.md) |
| Validation checklist | [`validation_checklist.md`](validation_checklist.md) |
| Scripts | `scripts/` (below) |
| Extracted data and new measurements | `data/`, `audit/` |

## Scripts

All run from the repository root with the `islvit` environment.

| Script | What it does | Needs | Output |
|---|---|---|---|
| `scripts/audit_results.py` | Recomputes every reported number from `runs/` and `splits/`, with its own McNemar implementation; rebuilds session blocks from file names | tracked files (clean-detection check also reads the local `cache128_f32_test/`) | `data/audit_results.json` |
| `scripts/measure_model.py` | Parameters, GMACs, model-only CPU latency of `release/..._s0.pt` | PyTorch | `data/model_measurements.json` |
| `scripts/make_figures.py` | Figures 5 to 9 from `data/audit_results.json` and run files | matplotlib | `latex/figures/` |
| `scripts/make_comparison_csv.py` | `model_comparison.csv` from the data files | — | `model_comparison.csv` |
| `scripts/app_integration_test.py` | Starts `islvit.demo`, posts held-out clips to `POST /predict`, compares with offline outcomes | local `INCLUDE_raw/`, model weights | `data/app_integration_<tag>.json` |
| `scripts/make_degraded_clips.py` | 180p / JPEG-40 lossless copies of the 50 app-test clips | local `INCLUDE_raw/` | local folder (not committed) |
| `scripts/run_crosscorpus_release.sh`, `relabel_cislr_split.py` | CISLR landmarks + scoring of the three release files | local CISLR videos and `cache_cislr_f32/` | `data/crosscorpus_release_s*.json` |
| `scripts/check_links.py` | Checks every hyperlink in the PDF (repository links against the pinned commit, others over HTTP) | git | `data/link_check.json` |

Commands used for the new measurements:

```bash
python report/final/scripts/audit_results.py
python report/final/scripts/measure_model.py
python report/final/scripts/app_integration_test.py --run release/isl_vit_tiny_lm_int4_s0.pt --tag int4_lm_s0 --offline runs/f16_262w_lm_s0_qat4/masked50_clean.json
python report/final/scripts/app_integration_test.py --run runs/f16_262w_lm_s0 --tag fp32_lm_s0 --offline runs/f16_262w_lm_s0/masked50_clean.json
python report/final/scripts/app_integration_test.py --run runs/f16_262w_s0 --tag fp32_pixel_s0 --offline runs/f16_262w_s0/masked50_clean.json
python report/final/scripts/app_integration_test.py --run release/isl_vit_tiny_lm_int4_s0.pt --tag int4_lm_s0_cpu --device cpu --limit 10 --offline runs/f16_262w_lm_s0_qat4/masked50_clean.json
python report/final/scripts/make_degraded_clips.py scratch_degraded180
python report/final/scripts/app_integration_test.py --run release/isl_vit_tiny_lm_int4_s0.pt --tag int4_lm_s0_180p_srl1 --sr runs/sr_l1/generator.pt --video-dir scratch_degraded180 --offline runs/gan_study/frozen_int4/L3_srl1_s0.json
bash report/final/scripts/run_crosscorpus_release.sh
python report/final/scripts/make_figures.py
python report/final/scripts/make_comparison_csv.py
bash report/final/latex/build.sh
python report/final/scripts/check_links.py report/final/latex/main.pdf
```

`audit/rescore_release_s{0,1,2}.json` are `python -m islvit.mask50 --run release/isl_vit_tiny_lm_int4_s*.pt` on the clean test cache (`ISLVIT_CACHE=cache128_f32_test ISLVIT_CACHE_LM=cache_lm_f32_test`); their per-clip outcomes equal the tracked `masked50_clean.json` files.

## Template

Official Springer Nature LaTeX template, version 3.1 (December 2024), downloaded on 2026-10-23 from `https://resource-cms.springernature.com/springer-cms/rest/v1/content/18782940/data/v12` (zip SHA-256 `812e76dcaa9c28dc1bff1fb6065d51729b67d4ea140552a05088317414a3ecae`). Class `sn-jnl.cls`, reference style `sn-mathphys-num`. Wide tables use the class's `sidewaystable`. Build with `pdflatex`, `bibtex`, `pdflatex`, `pdflatex` (`latex/build.sh`); tested with MiKTeX 25.12.

## Not committed

Local caches, model checkpoints under `runs/*/best.pt`, generator weights, source videos, the CISLR landmark cache `cache_lm_cislr_f32/` and the 180p clip copies are not in git, consistent with the rest of the repository.
