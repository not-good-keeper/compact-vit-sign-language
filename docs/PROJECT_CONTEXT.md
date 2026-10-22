# ISL Reader / ISL-ViT-Tiny: Project Context Document

> A single, self-contained account of everything in this repository: what was built, how it was built, which file does what, where every result lives, and which algorithm produced it and why. It is written so that someone who has never seen the project can find, rebuild and defend any number in it.

| | |
|---|---|
| Repository | `D:\nn_dl_prac` (GitHub: `not-good-keeper/compact-vit-sign-language`, branch `main`) |
| State described | `main` at commit `fcc2165` (2026-10-13), plus the working tree on the same day |
| Course | 23CSE473 Neural Networks and Deep Learning, Amrita School of Computing, Amrita Vishwa Vidyapeetham, Coimbatore |
| Team 3 | Yalamanchi Kushal (CB.SC.U4CSE23766), Nethi Kushala Kumar (CB.SC.U4CSE23735), Dhruseek Rishi Menon (CB.SC.U4CSE23716) |
| Faculty guide | Dr. T. Bagyammal |
| Project started | July 2026 (first commit 2026-09-02) |

---

## Contents

0. [How to read this document](#0-how-to-read-this-document)
1. [The project in one page](#1-the-project-in-one-page)
2. [Repository map and environment](#2-repository-map-and-environment)
3. [Data, corpora and split files](#3-data-corpora-and-split-files)
4. [Evaluation methodology](#4-evaluation-methodology)
5. [The data pipeline: video to crops and landmarks](#5-the-data-pipeline-video-to-crops-and-landmarks)
6. [The model: ISL-ViT-Tiny](#6-the-model-isl-vit-tiny)
7. [Training, pretraining, compression and release](#7-training-pretraining-compression-and-release)
8. [Research log: every experiment, in order](#8-research-log-every-experiment-in-order)
9. [The super-resolution GAN study](#9-the-super-resolution-gan-study)
10. [The application: server, interface, demo](#10-the-application-server-interface-demo)
11. [Documents and deliverables](#11-documents-and-deliverables)
12. [Command cookbook](#12-command-cookbook)
13. [Master results index](#13-master-results-index)
14. [Caveats, corrections and open gaps](#14-caveats-corrections-and-open-gaps)
15. [Glossary](#15-glossary)

---

## 0. How to read this document

### 0.1 Conventions

* **Paths** are relative to the repository root `D:\nn_dl_prac` unless they begin with `C:/` (the off-repo GAN work directory, §2.4). Forward slashes are used throughout.
* **Line numbers** (`file.py:123`) are as of commit `fcc2165` and will drift as code changes; the function name is the stable handle.
* **A "result" is always given with four things:** the value, the algorithm that produced it, why that algorithm was chosen, and where the number can be read. The last is a file path plus, for JSON, the key.
* **Evidence tags** say how well a number is backed inside the repository:

| Tag | Meaning |
|---|---|
| `[file]` | A machine-readable result (JSON/CSV) tracked in git. Re-reading it reproduces the number. |
| `[log]` | A tracked console log in the repository root (`*.log`). Human-readable output of the script named beside it. |
| `[local]` | Exists on this computer only (caches, checkpoints, demo clips, the GAN work directory). Gitignored or untracked. Regenerable with the command given. |
| `[report]` | Stated in a project document, but no committed result file or script reproduces it. Treat as a documented claim, not a reproducible one. §14 lists them. |

### 0.2 Metric definitions used everywhere

| Term | Definition | Code |
|---|---|---|
| **top-1 / top-5** | Share of clips whose true word is the highest-ranked / among the five highest-ranked outputs. | `islvit/tta.py:65 score` |
| **balanced accuracy** | Mean over words of per-word recall; unaffected by how many test clips each word has. | `islvit/tta.py:65 score` |
| **masked-50** | The deployed metric. The model keeps all 262 output columns; the 212 words outside the deployed 50 are set to zero probability before the top word is taken. | `islvit/mask50.py:56 evaluate_masked` |
| **6-view TTA** | Probabilities (not logits) averaged over three temporal phases (0.25, 0.5, 0.75 of each sampling segment) with and without a horizontal flip that swaps the two hand streams. | `islvit/tta.py:42-43,46`, `islvit/mask50.py:56` |
| **session-disjoint** | Whole recording sessions held out for test, so no signer, room, outfit or lighting is shared with training. | `islvit/data/splits.py:209 split_by_session` |
| **McNemar** | Exact two-sided paired test on the clips two models classify differently. | `islvit/gan_report.py` (`mcnemar`, `compare`) |
| **seed mean** | Every headline figure is the mean of 3 independently trained seeds; the best seed is never quoted. | rule stated in `islvit/release.py`, commit `a5fef61` |

### 0.3 Units

Older documents write "MB" for sizes that are actually mebibytes (2^20 bytes), because they were computed as bytes ÷ 1,048,576. The shipped model is **2,069,700 bytes = 1.974 MiB = 2.07 MB (decimal)**. This document says MiB where it matters and quotes bytes for the shipped file.

---

## 1. The project in one page

### 1.1 What the project is

**ISL Reader** recognises isolated **Indian Sign Language (ISL)** words from short video clips. The recogniser is **ISL-ViT-Tiny**, a 3.89 M-parameter *factorised* Vision Transformer. It reads three 64×64 crops per frame (left hand, right hand, face), 16 frames per clip, plus 21 hand-joint landmarks per hand, and names the word. It is sized for smart glasses: the shipped model is a 2,069,700-byte file.

The project has four connected parts:

1. **The classifier**: architecture, training recipe, compression to 4 bits (§6, §7).
2. **An honest evaluation**: the standard INCLUDE benchmark is dominated by near-duplicate recordings, so every result here is measured on whole recording sessions the model never saw (§3, §4).
3. **A super-resolution GAN front end** that replaces bicubic upscaling for low-resolution video, with a controlled GAN-versus-no-GAN study of three GAN versions (§9).
4. **An application**: a seven-screen web interface served by a Python server, with a demo launcher and ten built-in clips (§10).

### 1.2 The goal, and what was actually achieved

| Target (set at the start) | Result | Where |
|---|---|---|
| ≥ 75 % top-1 on the 50 deployed words, on unseen sessions | **85.9 %** (mean of three seeds) | §1.3 row 1 |
| Model of about 2 MB | **2,069,700 bytes** | §1.3 row 2 |
| Evaluation that does not reward memorising a recording session | Session-disjoint protocol, with a measured ~28 point leakage in the standard split under a matched recipe | §4, §14.1 |
| A working application | Seven screens, ten demo clips, same answers as the offline evaluation on the clips checked | §10 |
| A GAN somewhere it can be measured, with a with/without comparison | SR front end; three GAN versions each compared with bicubic and with a no-GAN control | §9 |

### 1.3 Headline results

Each row: the value, the algorithm behind it, why, and where it lives. All accuracies are masked-50, 6-view TTA, on the **472 held-out test clips** of the 50 deployed words (`splits/_test472_paths.csv`).

| # | Result | Value | Algorithm | Why this algorithm | Evidence |
|---|---|---|---|---|---|
| 1 | Shipped model accuracy | **85.9 %** top-1 (seeds 84.5 / 87.5 / 85.8); top-5 **96.4 %** (96.4 / 95.8 / 97.0); balanced **85.8 %** | Landmark ISL-ViT-Tiny, 1,000 epochs, iSign masked-image-modelling init, then 300 epochs of INT4 quantisation-aware training, group 128 | The pixel-only model plateaued near 74 %; landmarks gave +12.8; 4 bits are needed for the 2 MB budget and QAT keeps the accuracy | `runs/f16_262w_lm_s{0,1,2}_qat4/masked50_clean.json` `[file]`, key `masked_tta.{top1,top5,balanced}` |
| 2 | Shipped model size | **2,069,700 bytes** per file; **3,894,214** parameters | 4-bit weights in groups of 128 with FP16 scales, packed two codes per byte | Per-group scales stop one outlier crushing a 768×192 matrix; costs ~1.6 % overhead | `release/isl_vit_tiny_lm_int4_s{0,1,2}.pt` `[file]`; `islvit/export.py:246 pack_int4` |
| 3 | Gain from the hand-landmark stream | pixel-only **73.7 %** → landmark FP32 **86.5 %** (+12.8). Paired: 76 / 83 / 88 clips gained against 18 / 21 / 26 lost, p = 1.2e-9 / 6.7e-10 / 4.6e-9 | 21 MediaPipe joints per hand, made position- and scale-invariant, added to each hand token through a zero-initialised MLP (91,008 parameters) | The pipeline already computed these joints to draw a box and then discarded them | `runs/f16_262w_s{0,1,2}/masked50_clean.json` vs `runs/f16_262w_lm_s{0,1,2}/masked50_clean.json` `[file]`, keys `masked_tta.top1`, `hits_masked_tta`; `islvit/models/isl_vit.py:312 landmark_features` |
| 4 | Cost of quantisation | FP32 **86.5** → post-training INT4 **83.4** → QAT INT4 **85.9**. QAT vs post-training, pooled over seeds: 77 clips gained, 41 lost, p = 0.0012 | Straight-through-estimator fake quantisation inside training, sharing one rounding function with the exporter | Rounding after training lost 3.1 points; training with the rounding in the loop lost 0.6 | `runs/f16_262w_lm_s*/ptq4_clean.json` and `runs/f16_262w_lm_s*_qat4/masked50_clean.json` `[file]`; `islvit/qat.py:50 Int4Faker` |
| 5 | Why the 50-word mask matters | Answering from all 262 words: **72.0 %** top-1, **92.7 %** top-5 (vs 85.9 / 96.4 masked) | Zero the 212 non-deployed columns, then argmax | A 50-word product cannot emit the other words; the mask reproduces that exactly | same files, key `wide_tta.{top1,top5}` `[file]`; `islvit/mask50.py` |
| 6 | Random split vs unseen sessions, same recipe | **98.5 %** (one seed, 200 clips) vs **70.1 %** (69.9 / 70.3, two seeds, 472 clips): about **28 points** | One 50-word pixel-only recipe (16 frames, 2,000 epochs, iSign init) trained on `vocab50__random-video` vs on `vocab50clean__session-disjoint` | Quantifies how much of the standard benchmark is session memorisation. (The 262-word model masked to 50 words reaches 75.6 % on the same 472 clips but is a different model and is not used for this comparison; see §14.1) | `runs/baseline_random_s0/summary.json` key `test.top1`; `runs/f16_clean_s0/summary.json`, `runs/f16_clean_s1/summary.json` key `test.top1` `[file]` |
| 7 | Accuracy trajectory on one test set | 51.7 → 59.7 → 70.1 → 75.6 → 85.9 % | (a) 250 → 2,000 epochs, (b) 8 → 16 frames from a 32-frame cache, (c) 262-word model masked to 50 with INT4 QAT, (d) landmarks | Each rung is a separate, controlled change; none is architectural except the landmark input | §8, rows R11 to R18 |
| 8 | Low-resolution video, 180p | bicubic **61.0 %** → learned super-resolution **75.4 to 76.0 %** (frozen INT4 model); GAN versions 75.4 / 75.5 / 75.4 vs no-GAN control 76.0 | Replace the bicubic resize before detection with a ×4 generator | Low-resolution wearable video degrades the hand detector, not just the pixels | `runs/gan_study/frozen_int4/L3_*_s*.json`, `runs/gan_study_v1/frozen_int4/` `[file]`; §9 |
| 9 | GAN vs no GAN | The three GANs passed the pre-registered test against their no-GAN control in **2 of 9** comparisons, both at 240p | Exact McNemar on identical clips, pooled over 3 seeds; "helps" needs a higher mean, all three seeds the same way, p < 0.05 | A GAN that beats bicubic proves only that learning to upscale helps; the control isolates the adversarial loss | `docs/GAN_COMPARATIVE_STUDY.md` §7.2; `islvit/gan_report.py` `compare()` |
| 10 | Where the remaining low-res error is | At 180p with SR-GAN v2: 10.4 points missing; ~7.7 from degraded landmarks, ~2.8 from degraded pixels | Mix clean and degraded inputs (2×2 decomposition) | Separates what the generator should fix from what it cannot | `runs/gan_study/diagnostics/res_clean_pix_L3_*` and `.../mirror/*` `[file]` |

### 1.4 Timeline

From `git log` (first commit to latest):

| Date | Commit | What it contains |
|---|---|---|
| 2026-09-02 | `73c2caa`, `ab5bfbd` | Factorised ViT, crop pipeline, four split protocols, leakage analysis, report |
| 2026-09-09 | `a71cde5`, `0a34f11`, `0cc142b` | 32-frame cache, INT8 export, distillation harness, frame sweep, three failed levers, review deliverables |
| 2026-09-18 | `5fa43f3` | 2 MB budget met by one INT4 QAT model at 75.6 % |
| 2026-09-24 to 09-28 | `6cf84de`, `edd6ddf`, `cab93b4`, `594c537`, `c6629c3`, `160f31e` | Landmark stream (86 %), deterministic MediaPipe, clean test cache, honest INT4 QAT, third seed |
| 2026-09-29 to 09-30 | `d2b95f0`, `42080f0`, `8d0bc41` | Validation-based screening of landmark levers; none survives |
| 2026-10-08 | `a5fef61`, `b1fd1f2` | Shipped INT4 models in `release/`, loader, CPU path, case study |
| 2026-10-12 | `10b9d4a`, `cc90912`, `5f7d9cf`, `2c392fe` | SR-GAN front end and study, seven-screen interface and demo launcher, Review 3 report and slides |
| 2026-10-13 | `d930605`, `fcc2165` | Rewritten GAN study; revised report and slides; team details |

---

## 2. Repository map and environment

### 2.1 Top level

`[T]` = tracked in git, `[L]` = local only (gitignored or untracked).

| Path | What it is | Tracked |
|---|---|---|
| `README.md` | Public summary, headline table, layout | `[T]` |
| `islvit/` | All source code (§2.2) | `[T]` |
| `configs/` | YAML training configs, one per experiment family (§7.1) | `[T]` |
| `splits/` | 22 split definitions (CSV): the four protocols, vocabulary tiers, cross-corpus, the 472-clip test list (§3.4) | `[T]` |
| `runs/` | 129 entries. Per-run `summary.json` + `history.csv`, per-model scoring JSON, the GAN study results. Weights (`*.pt`) excluded (§2.3) | `[T]` (JSON/CSV), `[L]` (weights) |
| `release/` | The three shipped INT4 models, 2,069,700 bytes each | `[T]` (the one `*.pt` exception) |
| `docs/` | Technical reports, GAN study, improvement plan, capture protocol, figures | `[T]` |
| `report/case_study/` | Review 2 case study (Markdown/HTML/DOCX/PDF) and its diagram builders | `[T]` |
| `report/review3/` | Review 3 case-study report, slide deck and their builders and assets | `[T]` |
| `run_*.sh` (39 files) | One shell script per experiment; the executable record of every investigation (§8) | `[T]` |
| `*.log` (root) | Console output of those scripts. 40 are tracked; 13 newer ones (`gan_*.log`, `demo*.log`, `hand_proxy.log`, `patches_v3.log`) are not | mixed |
| `ISL_ViT_Tiny_Presentation.html`, `ISL_ViT_Tiny_Review_Document.html`, `ISL_ViT_Tiny_Review_Pack.docx` | Earlier-review deliverables (Reviews 1 and 2) | `[T]` |
| `download_include_compressed.py`, `start_include_headless.ps1` | Original INCLUDE download helpers (the 480p transcode that caused the integrity incident, §3.3) | `[T]` |
| `test.py` | **Unrelated** Playwright snippet (sums numbers in a web table). Not part of the project | `[T]` |
| `INCLUDE_raw/`, `INCLUDE_480p/`, `CISLR/`, `ISL_CSLTR/`, `iSign_meta/` | Source corpora (~57 GB). Never redistributed | `[L]` |
| `cache128_f32/`, `cache128_f32_test/`, `cache_lm_f32/`, `cache_lm_f32_fresh/`, `cache_lm_f32_test/`, `cache_cislr_f32/` | Crop and landmark caches (§5.4) | `[L]` |
| `models/` | MediaPipe `holistic_landmarker.task` and `hand_landmarker.task` (21 MB) | `[L]` |
| `demo_samples/` | Ten held-out clips + posters + `manifest.json` for the demo (§10.4). INCLUDE test videos, so not redistributed | `[L]` |
| `custom/` | Newly recorded takes (`custom/<signer>/<word>/takeN.webm`), empty so far | `[L]` |
| `nndlvit/`, `__pycache__/`, `scratch/` | Virtual environment, bytecode, scratch images | `[L]` |

### 2.2 `islvit/`: the code, file by file

Function index with line numbers as of `fcc2165`. "Doc" files are covered in the section named.

**Model and training**

| File | Lines | Purpose | Key objects |
|---|---|---|---|
| `models/isl_vit.py` | 421 | The classifier (§6) | `Attention:34`, `DropPath:57`, `Block:72`, `SpatialEncoder:87`, `ISLViT:136`, `landmark_features:312`, `load_deit_tiny_weights:347`, `count_parameters:407` |
| `train.py` | 734 | Supervised training; fine-tuning; resume | `build_param_groups:49`, `cosine_schedule:68`, `resize_position_embeddings:75`, `mixup_batch:125`, `cutmix_batch:142`, `ModelEma:181`, `evaluate:210`, `main:244` |
| `pretrain.py` | 424 | SimMIM-style masked-crop pretraining (§7.2) | `PretrainCrops:59`, `MaskedPretrainer:182`, `patchify:173` |
| `eval.py` | 99 | Score any checkpoint or release file on any split | `load_run:27` |
| `tta.py` | 156 | Six-view test-time augmentation | `view_probabilities:46`, `score:65` |
| `mask50.py` | 138 | The deployed metric; 50-word masking | `deployed_columns:45`, `evaluate_masked:56` |
| `gate.py` | 123 | Accuracy-versus-coverage under confidence gating | `averaged_probabilities:42` |
| `qat.py` | 234 | INT4 quantisation-aware training (§7.4) | `Int4Faker:50`, `quantised_copy:88` |
| `export.py` | 379 | INT8/INT4 packing, size survey, release loader | `pack_int8:56`, `save_int4:98`, `load_int4:142`, `load_release:181`, `int4_targets:202`, `int4_codes:223`, `pack_int4:246` |
| `release.py` | 38 | Writes `release/*.pt` from a QAT run | `main:23` |
| `int4_eval.py` | 59 | Post-training INT4 baseline (nothing trained) | `main:39` |
| `narrow.py` | 185 | Deployed-objective fine-tune (rejected, §8 R17) | `main:53` |
| `distill.py` | 304 | Ensemble-to-student distillation harness (no result recorded) | `select_blocks:63`, `distillation_loss:100` |
| `ensemble_val.py` | 74 | Does averaging train-only models help on validation? (§8 R22) | script |
| `predict.py` | 170 | Point the model at a video file | `extract:42`, `extract_landmarks:60`, `views:81` |
| `walkthrough.py` | 258 | Trace one clip through every stage with real tensors | `trace:44`, `figure:143` |
| `slt.py` | 362 | Sign-to-English translation scaffolding (smoke-test only) | `SignTranslation:98`, `bleu:198` |

**Data**

| File | Lines | Purpose | Key objects |
|---|---|---|---|
| `data/splits.py` | 460 | The four split protocols and their leakage assertions (§3.5) | `assign_take_groups:118`, `split_by_take_group:139`, `global_session_blocks:170`, `split_by_session:209`, `split_by_video:302`, `verify:365` |
| `data/crops.py` | 820 | Frame sampling, MediaPipe, two-stage hand detection, crop cache (§5) | `read_sampled_frames:163`, `frontend_spec:207`, `pad_to_work_width:236`, `pose_roi:305`, `hand_box_in_roi:329`, `fill_gaps:362`, `reset_detectors:413`, `extract_clip:490` |
| `data/landmarks.py` | 220 | Hand and pose landmark cache aligned to the crop cache (§5.3) | `hand_points_in_roi:56`, `extract_landmarks:90` |
| `data/dataset.py` | 550 | Torch `Dataset`, all augmentation, landmark loading and alignment guard | `interpolate_hands:67`, `augment_landmarks:93`, `check_landmark_alignment:129`, `IncludeCrops:169`, `prepare_clip:364` |
| `data/fold_val.py` | 69 | Builds `*-foldval` splits (validation folded into train) | `main:31` |
| `data/build_vocab_tier.py` | 171 | Vocabulary-restricted splits | `main:39` |
| `data/build_crosscorpus.py` | 179 | INCLUDE-train / CISLR-test split | `main:45` |
| `data/merge_cislr.py` | 128 | Labelled-CISLR merge splits | `main:64` |
| `data/ingest_custom.py` | 161 | Signer-disjoint split from newly recorded clips | `main:71` |
| `data/isign_extract.py` | 181 | Disk-bounded extraction from the split iSign zip | `ConcatFile:37`, `main:109` |
| `data/refetch_raw.py` | 152 | Restore 1080p sources deleted by the 480p transcode | `download:50`, `extract_videos:84` |

**Super-resolution GAN** (`sr/`, §9)

| File | Lines | Purpose |
|---|---|---|
| `sr/models.py` | 139 | `SRGenerator:32`, `UNetDiscriminatorSN:51`, `VGGPerceptualLoss:84` |
| `sr/degrade.py` | 149 | `degrade_frame:35` (study), `random_degrade:122` (training), `h264:101` |
| `sr/patches.py` | 156 | High-resolution patch banks with a leak guard |
| `sr/train.py` | 386 | Two-stage GAN training, v3 hand losses |
| `sr/handproxy.py` | 271 | Differentiable stand-in for MediaPipe's hand landmarker |
| `sr/upscaler.py` | 86 | Applies a trained generator in the frame reader; fp16 option |
| `sr/evaluate.py` | 127 | PSNR, SSIM, sharpness on held-out patches |
| `sr/study.py` | 135 | Path lists, guarded clean-up, detection and hallucination metrics |
| `sr/diag_detector.py`, `sr/diag_cislr.py`, `sr/diag_perclip.py` | 281 / 49 / 33 | Diagnostics (§9.7) |

**Application**

| File | Lines | Purpose |
|---|---|---|
| `serve.py` | 1,023 | Standard-library HTTP server: `/`, `/info`, `/predict`, `/correct`, `/enrol`; `classify:50`, `Handler:102`. **Owned by the user; never modified by the Review 3 work.** |
| `demo.py` | 205 | Launcher around `serve.py`: new UI, 50-word masking, `--sr`, host binding, `/samples` routes |
| `demo_samples.py` | 196 | Builds the ten demo clips and `manifest.json` |
| `ui/build.py` | 47 | Bundles `ui/src/` into one self-contained `ui/index.html` |
| `ui/src/` | 6,850 | `shell.html`, `base.css`, `core.js`, and seven screens (`capture`, `notsure`, `sentence`, `settings`, `diagnostics`, `vocabulary`, `enrolment`), each `.html`+`.css`+`.js` |

**Reporting**

| File | Purpose |
|---|---|
| `figures.py` | Figures 1 to 9 for the technical report (leakage ladder, convergence, resolution/frame ablations, box provenance, per-class recall) |
| `figures_v2.py` | Figures 10 to 20 for current results (leakage now, frame sweep, gating, waterfall, size frontier, detection quality, crop montage, training curves, per-class, size ladder) |
| `gan_report.py` | Generates `docs/GAN_COMPARATIVE_STUDY.md` and `.docx` and figures 21 to 23 from `runs/` |
| `report.py` | Tabulates finished runs |
| `make_social_card.py` | GitHub social preview image |

### 2.3 What is deliberately not in git

| Excluded | Size | How to get it back |
|---|---|---|
| Source corpora | ~57 GB | Download from the original sources (§3) |
| Crop caches | ~70 GB | `python -m islvit.data.crops ...` (§12) |
| Training checkpoints `runs/*/best.pt` | ~1 GB | Retrain with the `run_*.sh` script for that experiment |
| GAN generators `runs/sr*/generator.pt`, `runs/hand_proxy/proxy.pt` | small | Retrain with `run_gan_study.sh`, `run_gan_study_v2.sh`, `run_gan_v3.sh` |
| `demo_samples/` | 11 MB | `python -m islvit.demo_samples` |

Tracked and sufficient to **verify** every number: all code, all configs, all split CSVs, every `summary.json`/`history.csv`, every scoring JSON, the three release models.

### 2.4 The off-repo work directory `C:/isl_gan_work` (13 GB)

Everything the GAN study needs that is too large or too transient for git: high-resolution patch banks (`hr_train.npy`, `hr_train_v3.npy`, `hr_heldout.npy`, `hands_train_v3.npy`, `hp_train.npz`, `hp_val.npz`), temporary degraded extraction caches (`test_L*_*`, `cislr_*`), the validation-session development caches (`dev/`), and detector-lever caches (`diag/`). Set by `GAN_WORK` (default `C:/isl_gan_work`). `python -m islvit.sr.study drop` deletes only `crops.npy`, only inside this root, only from a directory carrying `FRONTEND.json`; it exists because earlier clean-up steps deleted 54 GB twice without checking.

### 2.5 Environment

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 3060, 12 GB |
| RAM | 32 GB; 12 CPU cores |
| OS | Windows 11 |
| Python | 3.11.15 (conda env `islvit`) |
| PyTorch | 2.6.0 + cu124 |
| MediaPipe | 1.0.0 (Tasks API) |
| timm | 1.0.28; OpenCV 5.0.0; NumPy 2.4.4; PyAV 18.0.0 (H.264 degradation) |
| Seeds | 0, 1, 2 (training), 1337 (split construction) |
| Shell | Git Bash; scripts use `/c/Users/ADMIN/anaconda3/envs/islvit/python.exe` |

Windows-specific facts that shaped the code: processes *spawn* rather than fork, so every command-line override must be forwarded into worker processes (`crops.py:386 init_worker` lists them all); and each MediaPipe worker holds a CUDA context, so worker counts are bounded by GPU memory.

---

## 3. Data, corpora and split files

### 3.1 INCLUDE (the labelled corpus)

INCLUDE (Sridhar et al., ACM Multimedia 2020; Zenodo record 4010759; CC-BY-4.0) is the only labelled corpus the classifier trains on.

| Property | Value |
|---|---|
| Clips used | 4,257 unique clips |
| Words | 262 word-level signs, 15 semantic categories |
| Clips per word | min 4, median 15, mean 16.2, max 22 (computed from `splits/full263__session-disjoint-foldval.csv`) |
| Recording | 1920×1080, 25 fps, one studio in Chennai, clips ≈ 2 to 5 s |
| Largest categories | Adjectives 788, People 513, Places 380, Home 379, Society 324 clips |
| Files on disk | `INCLUDE_raw/<Category>/<N. word>/MVI_xxxx.MOV` `[local]`; metadata parquets in `INCLUDE_raw/data/{train,val,test}-00000-of-00001.parquet` `[local]` |
| Capture id | Each filename carries the camera's running counter `MVI_xxxx`; the whole session structure is recovered from it (§3.5) |

The "median 15 clips per word across 262 words" is the defining difficulty: it is an extreme low-data regime for a Vision Transformer, and nearly every design choice in §7 follows from it.

### 3.2 The other corpora

| Corpus | Role | Facts | Where |
|---|---|---|---|
| **CISLR** v1.5-a (Joshi et al., EMNLP 2022) | Cross-corpus test and pretraining pool | Different signers, rooms, cameras; 300×300 px frames. 218 of INCLUDE's 262 words occur in it, covered by **609 test clips**. 6,096 clips passed the detection filter for pretraining | `CISLR/` `[local]`; `splits/crosscorpus__cislr-test.csv`, `splits/cislr__pretrain-allowed.csv` (6,745 rows) `[file]`; crop cache `cache_cislr_f32/` `[local]` |
| **iSign** v1.1 (Joshi et al., 2024) | Unlabelled self-supervised pretraining | 127,237 segments from 18,139 distinct source videos in a 54 GB two-part zip. **18,000 clips** were extracted, one per source video; **17,896** passed the hand-detection filter (genuine detection in at least 25 % of hand frames) and were used for pretraining (`isign_pipeline.log` `[log]`) | `iSign_meta/` `[local]`; extractor `islvit/data/isign_extract.py`; pretrained weights `runs/isign_mim/pretrained.pt` `[local]`. The iSign crop cache itself is no longer on disk. |
| **ISL-CSLTR** | Continuous-signing corpus, auxiliary only | Used for the translation scaffolding (`islvit/slt.py`); no result depends on it | `ISL_CSLTR/` `[local]`, `cslrt_download.log` `[log]` |
| **Custom recordings** | Planned signer-disjoint evaluation | Protocol in `docs/CAPTURE_PROTOCOL.md`: 3 signers × 50 words × 3 takes (~450 clips). The app's Enrolment screen writes to `custom/<signer>/<word>/takeN.webm`. No clips recorded yet | `custom/` `[local]`, `islvit/data/ingest_custom.py` |

### 3.3 A data-integrity incident, and its fix

An early run of `download_include_compressed.py` transcoded part of INCLUDE to 480p and **deleted the 1080p originals**. The loss was not random: 50 classes ended up entirely 480p and 212 entirely 1080p, with no class mixed (Adjectives 581 of 788 clips, Pronouns 168 of 168, Home 70 of 379). Encoding quality would have been a perfect predictor of those 50 labels, so a model could have learned compression artefacts instead of signs.

* **Fix:** `islvit/data/refetch_raw.py` re-downloaded nine archives (10.5 GB) from Zenodo, restoring 1080p for all 4,257 clips *before any reported experiment*.
* **Why it matters here:** it is the first of several places where a measurement looked fine and was not; the same habit (look at the data, not the metric) recurs in §5.2 and §8.
* **Evidence:** `refetch.log` `[log]`; technical report §4.4 `[report]`.

### 3.4 Split-file catalogue

All in `splits/`, all `[file]`. Columns: `video_path, label, parent_label, take_group, split` (+ `corpus` where more than one corpus is mixed). Counts are rows per `split` value, computed from the files.

| File | train / val / test (/ unused) | Words | Purpose |
|---|---|---|---|
| `full263__official.csv` | 3,101 / 340 / 816 | 262 | The shipped benchmark split, filtered to the 263-class benchmark |
| `full263__random-video.csv` | 2,843 / 542 / 872 | 262 | Naive per-clip random split (leaks takes) |
| `full263__take-group.csv` | 2,656 / 723 / 878 | 262 | No recording run straddles the boundary |
| `full263__take-group-foldval.csv` | 3,379 / – / 878 | 262 | As above with validation folded into train |
| `full263__session-disjoint.csv` | 1,957 / 888 / 1,010 (+402 unused) | 262 | **The strict protocol**; 888 validation clips share sessions with train |
| `full263__session-disjoint-foldval.csv` | **2,845** / – / **1,010** (+402 unused) | 262 | The training split of every final model: validation folded into train |
| `full263__session-disjoint+cislr.csv` | 2,450 / 888 / 1,010 (+402) | 262 | Adds 493 labelled CISLR clips to train (rejected lever) |
| `include50__official.csv` | 675 / 77 / 191 | 50 | INCLUDE-50 benchmark |
| `include50__random-video.csv` | 636 / 121 / 186 | 50 | INCLUDE-50, random |
| `include50__take-group.csv` | 624 / 160 / 159 | 50 | INCLUDE-50, take-group |
| `include50__session-disjoint.csv` | 512 / 166 / 252 (+13) | 50 | INCLUDE-50, session-disjoint (statistically underpowered, §4.4) |
| `vocab50clean__session-disjoint.csv` | 571 / – / **472** | 50 | **The deployed vocabulary with the deployed test set**; INCLUDE only |
| `vocab50full__session-disjoint.csv`, `vocab50+cislr__session-disjoint.csv` | 647 / – / 472 | 50 | As above plus 76 CISLR clips (the 76 hurt: −5.7, §8 R15) |
| `vocab30+cislr`, `vocab100+cislr`, `vocab137+cislr` `__session-disjoint.csv` | 404 / 1,289 / 1,750 train; test 278 / 865 / 976 | 30 / 100 / 137 | Vocabulary-size sweep tiers |
| `vocab50__random-video.csv` | 701 / 142 / **200** | 50 | The random-split control for the leakage comparison (§14.1) |
| `crosscorpus__cislr-test.csv` | 3,855 / – / **609** | 262 | Train on INCLUDE, test on CISLR |
| `crosscorpus__cislr-test-mix50.csv` | 3,149 / – / 305 (+1,010 unused) | 262 | Half the shared words contribute CISLR clips to train (304), half stay held out |
| `cislr__pretrain-allowed.csv` | 6,745 rows | – | CISLR clips allowed for self-supervised pretraining |
| `_test472_paths.csv` | 472 rows | 50 | The 472 deployed-word test clips as a path list. **Verified identical to the `test` rows of `vocab50clean__session-disjoint.csv` and a subset of the 1,010 test clips** |

### 3.5 How the splits are built, and what guards them

All in `islvit/data/splits.py` (460 lines). Seed 1337.

1. **Two benchmarks in one metadata file.** The HuggingFace parquets stack the 263-class split (`include_50 == False`, 4,257 clips) on the INCLUDE-50 split (943 clips). Ignoring the flag puts 277 clips in both train and test. `load_metadata:91` therefore sets `include_50` if a clip appears with that flag in *any* row.
2. **Take-groups** (`assign_take_groups:118`). Clips of one word whose capture ids are within `TAKE_GAP_THRESHOLD = 5` form one recording run (near-duplicate takes). The gap histogram is bimodal: 2,907 gaps of 1 or 2, only 19 in 3 to 19, then 1,063 of 20 or more; any threshold from 5 to 20 gives an identical partition (`docs/figures/fig01_mvi_gap_histogram.png`). `split_by_take_group:139` keeps each group wholly on one side.
3. **Session blocks** (`global_session_blocks:170`). Clustering capture ids over the **whole corpus** gives 13 blocks that behave like recording sessions. Clustering a subset instead gave 177 "blocks" that were really take-groups (§14.2): `reference=` must be the whole corpus.
4. **Session-disjoint split** (`split_by_session:209`). Random search over 20,000 assignments of blocks to test (each block chosen with probability 0.25); each candidate scored by (classes with ≥ 6 training clips minus twice the orphaned classes, minus starved test clips, minus size error against a 20 % test share). Validation is then carved from the *training* sessions at take-group granularity (a three-way session split left validation covering only 36 of 262 words). Test clips of words with fewer than 6 training clips are marked `unused` (402 clips).
5. **Foldval** (`data/fold_val.py`). Validation clips are folded into train for final models, giving 2,845 clips (+45 %). Reason: validation shares sessions with train, so it measures in-session recognition, not the cross-session recognition being reported. Selecting on it was worth −3.2 points compared with taking the final weights.
6. **Assertions** (`verify:365`, `report_session_split:333`): video-level, take-group and session-level disjointness are asserted at build time, the last against corpus-wide blocks. `build_crosscorpus.py:161-169` asserts the CISLR test clips appear in neither training rows nor the pretraining list and that no word has CISLR clips on both sides. `islvit/sr/patches.py` and `islvit/sr/handproxy.py` assert that no test-session or 472-clip video enters any GAN training bank.

---

## 4. Evaluation methodology

This section is the reason the project exists in its present form. Numbers in §8 are only meaningful with these rules in mind.

### 4.1 Why the standard benchmark is not trusted

INCLUDE was recorded as back-to-back takes. A random split puts takes 1 and 2 of a word in training and take 3 in test, so a model can score by recognising the session. Measured under the *same* recipe on the earlier 8-frame model (`docs/PRELIMINARY_MODEL_REPORT.md` §11.1, `[file]` via the runs named):

| Protocol | INCLUDE-263 top-1 | INCLUDE-50 top-1 | Runs `[file]` (`summary.json` key `test.top1`) |
|---|---|---|---|
| `random-video` | 94.5 % | 95.7 % | `runs/full263_v2__random-video`, `runs/include50_v2__random-video` |
| `official` | – | 95.3 % | `runs/include50_v2__official` |
| `take-group` | 29.3 % | 46.5 % | `runs/full263_v2__take-group`, `runs/abl64__take-group` |
| `session-disjoint` | 22.1 % | 14.3 % (31.3 % balanced) | `runs/sd_full263`, `runs/frames8__sd` |

About 65 points of the standard number is near-duplicate takes, a further ~7 is shared session conditions. The conventional protocol reproduces the published ~94 to 95 % and is mostly an artefact of the split.

### 4.2 The deployed metric, in detail

* **Train wide, deploy narrow.** The model has 262 outputs; the product answers from 50 words. `islvit/mask50.py:56` builds six views of every clip, averages their softmax probabilities, zeroes the 212 non-deployed columns, and takes the argmax. The deployed word list comes from `splits/vocab50clean__session-disjoint.csv` (`deployed_columns:45`), declared before any run and never influenced by test labels.
* **Why not train a 50-word model?** Measured on the identical 472 clips (§8 R10): a 50-word specialist scored 73.3 %, the 262-word model masked to 50 scored 75.6 %. The extra 212 words are five times the data and act as negatives.
* **Why TTA uses probabilities, not logits:** views and seeds are alternative observations of one clip, so a confidently right view should outvote an undecided one (`islvit/tta.py`, `islvit/gate.py` docstrings).
* **TTA is cheap and reliable** because it compares one checkpoint with itself, so seed variance cancels. +1.6 to +3.7 top-1 (mean +2.5) across eight checkpoints (`[report]`, technical report §11.4; each checkpoint's `summary.json` holds `test` and `test_tta` `[file]`).

### 4.3 Statistics, noise floor and decision rules

| Rule | Detail | Where |
|---|---|---|
| **Seed mean, never best seed** | Choosing the best seed would be selection on the test set. Seed 0 is only the default model by convention | `islvit/release.py`; commit `a5fef61` |
| **Measured noise floor** | Five seeds of one configuration (INCLUDE-263 session-disjoint): top-1 **35.0 ± 1.9 %** (range 33.2 to 38.0), balanced 38.4 ± 1.5, with TTA 37.5 ± 2.2. The difference between two single runs therefore has sd ≈ 2.7 points; claims under ~5 points from single runs are unresolved | `runs/sd263_ssl_foldval{,_s1,_s2,_s3,_s4}/summary.json` `[file]`, keys `test`, `test_tta` (recomputed for this document) |
| **McNemar exact test** | Paired comparison on identical clips, only discordant clips carry information. Holds seed fixed, so it cannot see run-to-run variation; the noise floor answers "does the intervention help?" | `islvit/gan_report.py` `mcnemar`, `compare` |
| **Pre-registered rule for the GAN study** | "Helps" needs: higher mean, all three seeds the same direction, pooled p < 0.05. "Hurts" is the mirror; anything else is "no reliable difference" | fixed before the first test run; `islvit/gan_report.py` `compare` |
| **Fixed budgets, final weights** | Epoch/iteration counts and loss weights fixed in advance; the final epoch or final EMA weights are kept. No checkpoint is chosen by test accuracy | `--select last` in `islvit/train.py`; `selection` field in every GAN `summary.json` |
| **Development on validation only** | From the landmark screening onward, all lever choices were made on the 888 validation clips (154 deployed-word clips), never on the 472 test clips; the winning recipe is retrained once and scored once | commit `d2b95f0`; `run_dev_screen.sh` |
| **A second seed before any claim** | Four separate claims in the project were overturned by a controlled follow-up (§8 R6, R7, R12, R14) | technical report §11.4, §11.5, §11.10, §11.11 |

### 4.4 A strict split can still be unusable

INCLUDE-50 `session-disjoint` is leak-free but statistically underpowered: 252 test clips over 45 words, with correlation **−0.73** between a word's test count and its training count (the best-represented test words are the least-trained). Top-1 there is 14.3 % but balanced accuracy 31.3 %; a frame-count ablation on it returned a U-shape (T = 4 best, T = 8 and 12 worse, T = 16 best) with no possible mechanism, and repeating on INCLUDE-263 made the order monotonic with T = 4 worst by 11 points. The decision rule that followed: refuse to read an incoherent ablation as a result. Evidence: `runs/frames{4,8,12,16}__sd` and `runs/sd263_f4`, `runs/sd263_f16` `[file]`; `fig09_temporal_ablation.png`.

### 4.5 Confidence gating (accuracy at coverage)

A wearable need not answer every clip. `islvit/gate.py` averages TTA probabilities across models, then reports accuracy over only the clips whose top probability is at least a threshold.

* The Settings screen's curve is `runs/gate_mixed.json` `[file]`: three pixel-only models (`f16_clean_s0`, `f16_clean_s1`, `f32_clean_s0`), 75.0 % at coverage 100 %, **88.7 % accuracy at τ = 0.4 with 75.2 % coverage**, 95.6 % at 48.5 % coverage (τ = 0.7).
* This curve is from an **older pixel-only ensemble**; it has not been re-measured for the shipped landmark model, and the interface labels it as such (§10.3).
* Thresholds are chosen on test, so these accuracies are optimistic as forward-looking estimates; the curve characterises the model, it is not a validated operating point.
* Off-domain behaviour: not one of 609 CISLR clips exceeded confidence 0.4, against 26.5 % of in-domain clips, so the model fails silently rather than confidently wrong (`[report]`, technical report §11.6; `crosscorpus_eval.log` `[log]` holds the accuracies).

---

## 5. The data pipeline: video to crops and landmarks

One path serves training caches, evaluation caches, the command-line predictor and the live server: `islvit/data/crops.py::extract_clip:490` and `islvit/data/landmarks.py::extract_landmarks:90`, both reached through `islvit/predict.py`. Writing a separate inference preprocessor is the standard way a model that scores well offline fails on a real clip, so there is not one.

```
video ──► read_sampled_frames ──► front end (degrade? / SR or bicubic?) ──► letterbox 1280×720
      ──► reset detectors ──► MediaPipe Holistic per frame
              hand found? ── yes ─► box from landmark hull × 1.5          (source = direct)
                          └─ no ─► ROI from pose forearm × 2.8, upscale to 256 px,
                                    HandLandmarker ── found ─► box        (source = roi)
                                                    └ no ─► carry nearest real box (source = interp)
      ──► crop to square, resize to 128 px  +  box geometry  +  21 joints/hand + 7 pose points
```

### 5.1 Frame sampling and the front end

`crops.py:163 read_sampled_frames` decodes the whole clip (50 to 120 frames, so sequential reading beats seeking) and takes `count` frames at `np.linspace(0, len-1, count).round()`; the caches use **32 frames**, the model samples **16** from them.

* **Working resolution.** Every frame is resized to height 720 and letterboxed to 1280 wide (`pad_to_work_width:236`, never cropped, so a hand near the edge is kept). Fixed height *and* width exist because MediaPipe's holistic graph keeps state and hard-fails when consecutive frames differ in size; a first fix that fixed only the height still crashed on iSign's varied aspect ratios.
* **Front end (the GAN's insertion point).** If the decoded height is not 720, the frame is brought to it: `INTER_AREA` when shrinking, `INTER_CUBIC` (bicubic) when enlarging, **or the super-resolution generator** when `UPSCALER` is set and the scale is above 1 (`crops.py:195-199`). 1080p video is shrunk and never reaches the generator, so results on clean INCLUDE cannot change; crops from clean clips with and without `--sr` were verified byte-identical `[report]`.
* **Controlled degradation for the study** (`DEGRADE`, `crops.py:187`): an area downscale plus one JPEG encode, deterministic, applied before anything else (`islvit/sr/degrade.py:35 degrade_frame`).
* **Provenance of the front end.** `crops.py:207 frontend_spec` records the degradation, generator path, generator SHA-256 and (if on) `sr_precision: fp16` in `FRONTEND.json` beside every cache; `dataset.py:129 check_landmark_alignment` refuses a crop cache and landmark cache built through different front ends.

### 5.2 Hand and face boxes: the two-stage detector

| Stage | Settings | Code |
|---|---|---|
| 1. Holistic landmarker, IMAGE mode | `min_hand_landmarks_confidence=0.3`, segmentation off. Box = square around the landmark hull × `HAND_MARGIN` 1.5 (face × `FACE_MARGIN` 1.3) | `crops.py:431 get_holistic`, `box_from_points:272` |
| 2. ROI rescue when stage 1 misses a hand | ROI centred on the pose wrist/thumb/index/pinky points, side = 2.8 × forearm length (min 60 px), pose visibility ≥ 0.3; crop upscaled to 256 px; dedicated `HandLandmarker` with permissive thresholds 0.2 | `crops.py:305 pose_roi`, `329 hand_box_in_roi`, `456 get_hand_landmarker` |
| 3. Still missing | Carry the nearest real detection into the gap, mark the crop `interp` | `crops.py:362 fill_gaps` |

Why two stages: full-frame Holistic found a hand in only ~79 % of frames (a hand is ~190 px in a 1,920 px frame and blurs at 25 fps); stage 2 adds 10 to 15 %.

Measured provenance over all 4,257 × 16 frames (`[report]`, technical report §7.1, Table; also drawn in `docs/figures/fig06_box_sources.png`, `fig15_detection_quality.png`): left hand 81.4 % direct / 9.9 % ROI / 8.6 % interpolated; right hand 70.3 / 15.1 / 14.6; face 99.9 % direct. In the 32-frame cache: left hand genuine 91.1 %, right 85.2 %, face 99.9 %.

**A negative result that shaped the design.** An early version synthesised the fallback box directly from pose geometry. It reported 100 % coverage and "healthy" skin-tone pixels and was badly wrong: pose reports optimistic visibility for occluded hands, so crops were shirt fabric and wall. *Coverage measured box existence, not box correctness.* Rendering the crops and looking at them caught it, and it is why every crop now carries a provenance code and why interpolated boxes are flagged to the model (§6.3).

### 5.3 Landmarks

`landmarks.py` re-runs the identical detection path and stores, row-aligned to the crop cache:

| Array | dtype, shape | Meaning |
|---|---|---|
| `hands.npy` | float16 `(N, T, 2, 21, 3)` | left/right hand, 21 joints, x, y frame-normalised, z relative depth |
| `hand_src.npy` | uint8 `(N, T, 2)` | 0 missing, 1 holistic, 2 ROI |
| `pose.npy` | float16 `(N, T, 7, 3)` | nose, left/right shoulder, elbow, wrist: x, y, visibility |
| `done.npy` | bool `(N,)` | resume marker |
| `SOURCE_CACHE` | text | the crop cache it was built against and the frame count |

**Alignment is the whole contract.** Landmark *t* of row *r* must come from the same video frame as crop *t* of row *r*. Checked, not assumed: over the 472 held-out clips (23,957 directly detected hand-frames) the landmark hull centre matches the crop-box centre with median error 0.0004 of the frame `[report]`; the loader refuses a cache pair with a different source, frame count or front end (`dataset.py:129`).

### 5.4 Caches on disk

| Cache | Contents | Size | Built by |
|---|---|---|---|
| `cache128_f32/` | 4,257 clips × 32 frames × 3 streams × 128 px (`crops.npy` uint8, `sources.npy`, `geometry.npy` float16 `(N,32,3,3)`, `index.csv`, `done.npy`) | 19 GB; built in 98 min, 0 failures | `python -m islvit.data.crops --cache-dir cache128_f32 --crop-size 128 --frames 32` |
| `cache_lm_f32/` | Landmarks for the above | 39 MB | `python -m islvit.data.landmarks --cache cache128_f32 --out cache_lm_f32 --frames 32` |
| `cache128_f32_test/`, `cache_lm_f32_test/` | The **472 test clips re-extracted with a fresh detector per video**; what every headline number is scored on | 2.1 GB, 4.3 MB | `run_clean_test.sh` |
| `cache_lm_f32_fresh/` | Earlier fresh-detector landmark pass (`masked50_fresh.json`) | 39 MB | `run_clean_test.sh` era |
| `cache_cislr_f32/` | CISLR crops for the cross-corpus test and the detection study | 2.7 GB | `crops.py --corpus cislr` |

Why a **128 px cache for a 64 px model:** the extra resolution makes spatial augmentation (scale-jitter windows) and temporal augmentation free at train time (memmap indexing only), and a cleaner 128 → 64 area-downsample is worth +4.4 points over an 80 → 64 cache `[file]` (`runs/abl64__take-group` 46.5 vs `runs/include50_v2__take-group` 42.1 on the same 64 px input).

### 5.5 The stateful-detector defect and its fix

MediaPipe's detectors keep state between videos **even in IMAGE mode**. Processing the same clip twice in one process flipped hand presence on 3 to 6 % of frames and moved pose by up to 0.2; a freshly built detector reproduced exactly. Every cached clip's detections therefore depended on whichever clips its worker had handled before it.

* **Fix:** `crops.py:413 reset_detectors` rebuilds both graphs at the start of every video (~0.6 s), in both the crop and landmark paths; two runs of one clip then agree to 0.0.
* **Consequence handled:** the 472 test clips were re-extracted this way into a separate cache (`cache128_f32_test`) and all models re-scored. Nothing moved by more than 0.6 points (pixel-only 75.6 / 73.5 / 72.0 → 75.0 / 73.7 / 72.2; landmark 87.7 / 87.3 / – → 87.3 / 86.9 / 85.4), so no earlier conclusion was an artefact, but the clean set is the one quoted because it matches deployment.
* **Evidence:** `masked50.json` (history-dependent cache) vs `masked50_fresh.json` vs `masked50_clean.json` in each `runs/f16_262w*_s*/` `[file]`; `run_clean_test.sh`, `landmarks_fresh.log` `[log]`.

---

## 6. The model: ISL-ViT-Tiny

Code: `islvit/models/isl_vit.py` (421 lines). Parameter counts below were computed for this document by instantiating `ISLViT(n_classes=262, n_frames=16, landmarks=True)`.

### 6.1 Structure

```
Input  crops   (B, T=16, S=3, 3, 64, 64)    S = left hand, right hand, face
       detected (B, T, S) bool              real detection vs interpolated/missing
       geometry (B, T, S, 3)                box centre x, centre y, size (normalised)
       hands (B,T,2,21,3), hand_present (B,T,2), pose (B,T,7,3)     [landmark model only]
  │
  ├─ STAGE A  SpatialEncoder, weights shared across all 48 crops
  │     Conv2d 3→192, k=16, s=16  → 4×4 = 16 patches
  │     + CLS token + position embedding → 17 tokens × 192
  │     4 × pre-norm Block (dim 192, 3 heads, MLP 768, GELU, stochastic depth to 0.1)
  │     LayerNorm → take CLS            → one 192-d embedding per crop
  │
  ├─ TOKEN CONDITIONING  (added to each crop embedding)
  │     + stream embedding (3 × 192)   + time embedding (16 × 192)
  │     + geometry_proj  Linear(3→192)→GELU→Linear(192→192), last layer zero-initialised
  │     + missing_embed where the box is interpolated or missing
  │     + hand_proj(65-d handshape+wrist features) × hand_present      on each hand token
  │     + pose_proj(21-d body features) × pose_ok                     on the face token
  │
  ├─ STAGE B  temporal encoder: prepend CLS → 49 tokens × 192, 4 × Block, LayerNorm
  │
  └─ Linear 192 → 262
```

| Quantity | Value |
|---|---|
| Parameters, shipped landmark model | **3,894,214** |
| of which spatial encoder | 1,930,944 |
| of which hand + pose landmark projections | **91,008** (≈ 0.045 MB at INT4) |
| Parameters, pixel-only 262-way | 3,803,206 (50-way: 3.762 M) |
| Attention tokens | 17 within each crop, then 49 across the clip, against 48 × 17 = 816 for joint attention |
| FP32 size | 3.894 M × 4 B = 15.6 MB (14.9 MiB) |

### 6.2 Why each piece exists

| Decision | Reason | Evidence |
|---|---|---|
| **Factorised encoder** (spatial then temporal) rather than joint attention | Joint attention is O((T·S·P)²); factorising gives O(P²) + O((T·S)²), ≈ 0.02× the attention cost at T = 8, which is what makes an edge budget reachable | `islvit/models/isl_vit.py` docstring; technical report §2.3 `[report]` |
| **Crops, never full frames** | Removes the studio wall, furniture and clothing as class cues; recovers hand detail that is ~10 % of frame width; ≈ 10× fewer tokens | technical report §3.1 `[report]`; pipeline §5 |
| **Width pinned to 192, 3 heads** | So ImageNet DeiT-Tiny and the iSign SSL weights load with no projection; a narrower student would forfeit initialisations worth ≈ +6 and ≈ +10 points | `islvit/distill.py` docstring; §8 R4 (DeiT +6.3) |
| **Geometry re-injected** | Cropping deletes *where* the hand is, and location separates words in ISL (the same handshape at the forehead and the chest differ) | `isl_vit.py:174-177` |
| **Zero-initialised projections** (geometry, hand, pose) | At step 0 the model *is* the pixel-only model (logit difference 0.0), so the pretrained initialisation transfers untouched and location/handshape are learned gradually | `isl_vit.py:200-207` |
| **Missing/interpolated flag** | An interpolated box is a stale box showing background; feeding the crop while flagging it lets the network discount it, and "no hand visible" is itself informative (many signs are one-handed) | `isl_vit.py:170-173`; `dataset.py:310` |
| **Spatial grid 4×4, 64 px** | Tripling tokens (112 px) changed top-1 by 0.0 and 128 px was 2.5 points worse; the cheapest setting was the most accurate | `runs/abl{64,112,128}__take-group` `[file]` (46.5 / 46.5 / 44.0) |
| **16 frames** | Below 8 frames temporal sampling is a real bottleneck (−9.4); doubling 8 → 16 at the proper budget is +6.4 | §8 R3, R12 |

### 6.3 Landmark features (`landmark_features`, `isl_vit.py:312`)

Computed *inside the model* so no inference path can compute them differently.

* **Handshape (63 numbers per hand):** the 21 joints minus the wrist, divided by the hand's own extent (largest in-plane joint distance, clamped ≥ 1e-3). Independent of where the hand is and how far from the camera (verified invariant to 5e-7). The x axis is multiplied by 16/9 first, because frames are letterboxed to 1280×720 and an unscaled x unit is 16/9 of a y unit; without it, a hand rotated 90° would read as a different shape.
* **Sign location (2 numbers):** the wrist relative to the shoulder-midpoint, in shoulder widths, independent of framing.
* **Body pose (21 numbers):** nose, shoulders, elbows, wrists in the same body frame, plus visibility. `pose_ok` is true only if both shoulders have visibility > 0.3; otherwise every body-relative number is zeroed rather than divided by nothing.
* **Where they go:** the 65-d hand vector goes through `hand_proj` (Linear 65→192, GELU, Linear 192→192) and is **added** to that hand's crop token, gated by `hand_present`; pose goes through `pose_proj` onto the face token. Token count and the temporal stage are unchanged.
* **Augmentation reaches the joints too:** flip and stream-dropout use the same random draws as the pixels (`dataset.py:458-470, 500-513`), mixup mixes landmarks with the same λ (`train.py:125`).

Optional landmark variants exist behind flags and are **off in the shipped model** because validation screening rejected them (§8 R21): `lm_velocity` (per-hand frame-to-frame change), `lm_pair` (right-minus-left wrist vector and length), `lm_wrist_vel` (wrist motion), `--lm-interp` (fill gaps ≤ 4 frames), `--lm-aug` (rotate/scale/shear/noise on joints).

### 6.4 Complexity and latency

| Quantity | Value | Evidence |
|---|---|---|
| Compute | 0.824 GMACs per 8-frame clip; 1.65 GMACs at 16 frames | technical report §8.6 `[report]` |
| Transformer latency | ~12 ms per 8-frame clip, ~24 ms per view at 16 frames (desktop CPU) | `islvit/serve.py` docstring; report `[report]` |
| Detector cost | MediaPipe costs 30 to 60 ms per frame on CPU, 3 to 5× the transformer; it, not the ViT, is the inference bottleneck | technical report §16.3 `[report]` |
| End to end in the app | 5 to 9 s per clip on a desktop CPU with six views (7.4 s measured on the Capture screenshot) | `report/review3/assets/ui_capture.png`; Review 3 report Table 6 |

---

## 7. Training, pretraining, compression and release

### 7.1 Supervised recipe (the shipped model)

Config `configs/full263_v2.yaml` plus command-line overrides; the effective recipe is recorded in each run's `summary.json` key `recipe` `[file]`. Training script `islvit/train.py:244 main`.

| Setting | Value | Why |
|---|---|---|
| Data | `splits/full263__session-disjoint-foldval.csv`: 2,845 training clips, 262 words, cache `cache128_f32` | Folding validation in gave +7.7 (§8 R6) |
| Input | 16 frames sampled from the 32 cached, 64 px, patch 16 | §8 R12, R14 |
| Initialisation | `--init-from runs/isign_mim/pretrained.pt` (iSign SSL); position and time embeddings resized by `train.py:75 resize_position_embeddings` (bicubic on the patch grid, linear on time) | SSL init is worth ≈ +10; without the resize it silently falls back to ImageNet in any other geometry and ablations measure pretraining instead of capacity |
| Epochs | **1,000**, fixed; final epoch kept (`--select last`) | 250 → 2,000 epochs was worth +7.9; selecting on a validation set that shares sessions with train optimises the wrong quantity |
| Optimiser | AdamW, peak LR 5e-4, 10 warm-up epochs, cosine to 1 % (`cosine_schedule:68`), batch 64, weight decay 0.05 on weight matrices only | standard for ViT; no decay on norms/biases (`build_param_groups:49`) |
| Backbone LR | × 0.1 for the spatial encoder | refine pretrained features, do not erase them |
| Regularisation | label smoothing 0.1; mixup α 0.2 on 50 % of batches; stochastic depth 0.1; gradient clip 1.0 | ~16 clips per word memorise easily |
| EMA of weights | decay 0.99 with warm-up `min(decay, (1+t)/(10+t))` | a fixed 0.999 never left its random initialisation when an epoch is ~9 steps and sat at chance for whole runs (a silent bug, fixed) |
| Precision | bfloat16 autocast on GPU | |
| Augmentation | temporal jitter inside uniform segments; scale-jitter crop window 0.8 to 1.0 of the cache (centre 0.9 at test); horizontal flip p 0.5 **with the hand streams swapped and geometry and joints mirrored**; brightness/contrast ±0.4; per-channel gain ±0.4; greyscale p 0.15; stream dropout p 0.15; **resolution jitter p 0.5**, one factor U(0.35, 0.9) per clip (degrade to a lower resolution and back) | Appearance augmentation exists because the protocol holds out whole sessions, so lighting and signer appearance are what the model must not key on. Resolution jitter is one factor per clip, not per frame: a flickering sharpness is an artefact no camera produces |
| Seeds | 0, 1, 2 → `runs/f16_262w_lm_s{0,1,2}` | |

Run order that produced the shipped model: `run_landmarks.sh` (seeds 0, 1), `run_lm_s2_resume.sh` (seed 2, which crashed at epoch ~235 and was resumed from its checkpoint), `run_lm_qat.sh` (QAT for seeds 0, 1), `run_clean_test.sh` (clean test cache). Logs: `landmarks.log`, `lm_s2_resume.log`, `lm_qat.log`, `clean_test.log` `[log]`.

Safety rails in the training script: `--overfit-batch` (a single batch must reach 100 % in under 40 steps), `--resume` with a fingerprint check, `--track-test` (diagnostic only, never used for selection), `--val-every N` (selection sees only scored epochs).

### 7.2 Self-supervised pretraining

Code `islvit/pretrain.py` (424 lines); runs `runs/isign_mim`, `runs/isign_mim16`, `runs/cislr_mim`, `runs/cislr_mim_rj` (`summary.json` `[file]`; `pretrained.pt` `[local]`).

| Run | Corpus | Clips | Epochs | Final loss | Note |
|---|---|---|---|---|---|
| `isign_mim` | iSign | 17,896 | 100 | 0.351 | **The shipped model's initialisation.** Sampled 8 frames from a 16-frame cache |
| `isign_mim16` | iSign | 17,896 | 100 | 0.322 | 16 of 16 frames (no temporal jitter): lower loss, **worse transfer** (60.0 vs 69.9) |
| `cislr_mim` | CISLR | 6,096 | 100 | 0.335 | Fixed native blur |
| `cislr_mim_rj` | CISLR | 6,096 | 100 | 0.316 | With resolution jitter at pretrain time: rejected |

Algorithm: **SimMIM-style** masked reconstruction, not MAE: masked patches are replaced by a learned mask token *inside* the encoder rather than dropped (with only 16 patches per crop, dropping tokens leaves 4 visible and a ragged batch, and substituting keeps the encoder identical to the one later fine-tuned). Mask ratio **0.75**, sampled globally across time, stream and space (masking within one crop would let the spatial encoder solve the task alone and leave Stage B untrained). A shallow decoder (dim 128, depth 2, 4 heads) predicts per-patch-normalised pixels; loss is mean squared error on masked patches.

**The loss is gated on genuine detection.** CISLR has a median of only 40.6 % genuinely detected hand frames per clip against ~88 % on INCLUDE, so 38 to 52 % of boxes are interpolated stale boxes; reconstructing them teaches the model to predict background. Interpolated and missing crops are excluded from the loss (`min_genuine 0.25` also filters clips).

### 7.3 Fine-tuning a trained model to a new front end

`train.py --finetune-from <best.pt>` loads the full model including the head, then trains at `--lr 1e-4` for a fixed 150 epochs with resolution jitter and the final epoch kept. Used for the retrained arm of the GAN study (`runs/ft360_{bicubic,srgan_v1,srgan}_s{0,1,2}`, §9.5). (An earlier `--init-from` only accepted an SSL `"backbone"` and dropped `head.*`.)

### 7.4 Compression: from 14.9 MiB to 1.97 MiB

All accuracies masked-50 TTA, 472 clips. The ladder below is the landmark model (`[file]`: `runs/f16_262w_lm_s*/{masked50_clean,ptq4_clean}.json`, `runs/f16_262w_lm_s*_qat4/masked50_clean.json`).

| Representation | Size | Top-1 (3-seed mean) |
|---|---|---|
| FP32 landmark model | ~14.9 MiB | 86.5 % |
| INT4, post-training (`islvit/int4_eval.py`) | ≈ 2 MB | 83.4 % |
| **INT4, quantisation-aware (shipped)** | **2,069,700 B** | **85.9 %** |

(For the earlier pixel-only model the ladder was 18.5 MB five-model ensemble 75.8 % → 14.55 MB single model 75.6 % → packed INT8 3.74 MB 75.6 % → INT4 post-training 1.95 MB 74.4 % → INT4 QAT 1.95 MB 75.6 %; `[report]` §11.12. The 0.2-point ensemble advantage was one clip in 472 and the ensemble was withdrawn as the headline.)

**Why 4 bits.** INT8 over the Linear layers leaves a 3.76 M model at 3.70 MB, nearly twice the budget. Halving the *bit width* costs accuracy that can be measured in minutes; halving the *architecture* costs accuracy nobody had measured, so bits were tried first (`islvit/export.py` `pack_int4` docstring).

**The algorithm** (`export.py:223 int4_codes`, `:246 pack_int4`):

1. Quantise each weight matrix in **groups of 128** consecutive weights.
2. Per group: `scale = max|w| / 7`, `code = clamp(round(w / scale), −8, 7)` (symmetric).
3. Store two 4-bit codes per byte and one **FP16 scale** per group (≈ 1.6 % overhead). A single scale across a whole 768×192 matrix would be dominated by its largest outlier.
4. Keep embeddings, `cls_token`, norms and biases at FP16 (`int4_targets:202` selects module weights with ndim ≥ 2 only). Together they are under 4 k parameters and the most perturbation-sensitive tensors.
5. The scale is rounded to FP16 *before* reconstruction so the model that is scored is the model that is saved.

**Quantisation-aware training** (`qat.py`): 300 epochs at LR 5e-5, group 128, EMA weights, **final epoch kept, fixed in advance** (an earlier QAT run chose its best epoch by test accuracy and its 75.6 % was withdrawn as optimistically biased; the landmark model's QAT was re-run in the corrected form). `Int4Faker:50` performs a straight-through estimator **by swapping tensors**: stash the master weights, overwrite the parameters with their 4-bit reconstruction, forward and backward, restore. (`torch.nn.utils.parametrize`, the idiomatic route, breaks under the `copy.deepcopy` inside `ModelEma`.)

Three things make QAT correct rather than approximately correct, two of which were bugs first:

* QAT and the exporter must round identically. They originally did not (the exporter caught `cls_token`, `stream_embed` and `time_embed` by dimension; QAT looked for a parameter named `weight`), the one failure that makes QAT silently worthless while still producing plausible numbers. Both now call `int4_targets` and `int4_codes`; the sets were verified identical.
* Embeddings stay full precision in both paths.
* Verification that the scored file is the shipped file: re-quantising a saved checkpoint moved no weight by more than 5.96e-8 (float32 epsilon) `[report]`.

**Cost of 4 bits.** Top-5 of the pixel-only model fell from 94.7 % to 93.2 % while top-1 held: four bits blunt the tail of the distribution even where the argmax survives, which matters for the interface's ranked alternatives and for confidence gating.

### 7.5 The release files

`islvit/release.py` writes `release/isl_vit_tiny_lm_int4_s{0,1,2}.pt` from the three QAT runs in the compact `export.save_int4` container (4-bit nibbles, FP16 scales, leftovers, plus the config and the word list, so each file is a complete deliverable; per-tensor `torch.save` overhead that pushed a 2,004 KiB model into a 2,052 KiB file is gone). Each file is **2,069,700 bytes (1.974 MiB)**. `islvit.export.load_release` reads them and `islvit.eval.load_run` accepts them, so `predict`, `serve`, `mask50` and `demo` all take `--run release/<file>.pt`.

* **Seed 0 is the default by convention, not by score**; choosing the best seed would be selection on test. The expected accuracy is the three-seed mean, 85.9 %.
* **Verified, not assumed:** loading each release file and scoring the clean test set reproduces the measured checkpoints exactly (84.5 / 87.5 / 85.8 %), so the FP16-scale reconstruction loses nothing (commit `a5fef61`, message).
* Fixed while shipping: `torch.autocast("cuda", enabled=False)` still queries the GPU on construction, so `predict` and `serve` crashed on machines without a usable GPU; off-GPU they now skip autocast.

### 7.6 Things built and not shipped

| Item | Status | Why |
|---|---|---|
| `islvit/narrow.py` (fine-tune the 262-word model on the 50-word softmax) | Rejected | Seeds moved in opposite directions (+2.8 p = 0.019; −0.4 p = 0.85; −3.6 p = 0.009): mean 73.3 vs 73.7. `runs/f16_262w_s{0,1,2}_narrow` `[file]` |
| `islvit/distill.py` (student with 2+2 blocks, 192 wide, ≈ 1.95 M parameters, learning an ensemble's output distribution) | Built, no run recorded | Superseded by QAT meeting both targets with a single model |
| Five-model ensemble | Not shipped | 18.5 MB; its edge over one model was inside noise |
| `islvit/slt.py` (ISL video → English text) | Scaffolding only | iSign's own paper reports BLEU-4 0.56; any BLEU here is a smoke test, not a quality claim |

---

## 8. Research log: every experiment, in order

Each entry gives the question, the method (script and run names), the result, the verdict, and the evidence. Numbers were checked against the run files named. Where the project **reversed itself**, the reversal is recorded, because the corrected path is more informative than the final state.

A note on the numbers: before the 32-frame cache, runs used the 8-frame model at 250 epochs; those absolute levels are *pessimistic* (§8 R11 shows every early run was undertrained), but comparisons within one table remain valid since each table's runs share a budget.

### R1. How much of the standard benchmark is leakage?

* **Question:** does the published ~94 to 95 % measure sign recognition?
* **Method:** one recipe, four split files (`splits/*__{official,random-video,take-group,session-disjoint}.csv`), trained and scored identically. `run_phase2.sh`.
* **Result:** see §4.1. Training loss curves are indistinguishable across protocols (all fall from 3.9 to ≈ 0.7), so optimisation is not the cause; the difference is **what the test set contains**. Leakage decomposes into ~63 points of near-duplicate takes and ~7 of shared sessions (balanced accuracy: take-group step −62.7 on INCLUDE-263, session step −6.8).
* **Verdict:** adopt session-disjoint as the protocol of record.
* **Evidence:** `runs/{full263_v2,include50_v2}__*`, `runs/sd_full263`, `runs/frames8__sd` `[file]`; `phase2b.log`, `phase2c.log`, `session_disjoint.log` `[log]`; `docs/figures/fig02_leakage_ladder.png`, `fig03/fig04` (convergence).

### R2. Does more spatial resolution help?

* **Method:** `run_ablation_res.sh`. 64 / 112 / 128 px input on the 128 px cache, INCLUDE-50 take-group, same recipe.
* **Result:** top-1 **46.5 / 46.5 / 44.0 %** at 16 / 49 / 64 tokens per crop and 0.824 / 2.342 / 3.010 GMACs.
* **Verdict:** tokens are not the bottleneck; 64 px kept. More tokens only give more ways to overfit ~600 clips.
* **Evidence:** `runs/abl{64,112,128}__take-group/summary.json` `[file]` (`test.top1` 0.4654 / 0.4654 / 0.4403); `ablation_res.log` `[log]`; `fig05_resolution_ablation.png`.

### R3. Does more temporal sampling help? (and a measurement lesson)

* **Method:** `run_ablation_frames.sh` (INCLUDE-50 session-disjoint) and `run_frames_full263.sh` (INCLUDE-263 session-disjoint), 4/8/12/16 frames from the 16-frame cache.
* **Result:**

| Frames | INCLUDE-50 SD top-1 | INCLUDE-263 SD top-1 |
|---|---|---|
| 4 | 21.8 % | 12.7 % |
| 8 | 14.3 % | 22.1 % |
| 12 | 12.3 % | – |
| 16 | 22.2 % | 22.7 % |

* **Verdict:** INCLUDE-50's order is a U-shape with no mechanism, so it was read as **instrument noise** (§4.4) and no decision was taken from it. INCLUDE-263 is monotone: T = 4 costs 9.4 points, T = 16 buys 0.6 at that budget. (R12 later shows the 250-epoch budget hid a large frame effect.)
* **Evidence:** `runs/frames{4,8,12,16}__sd`, `runs/sd263_f4`, `runs/sd_full263`, `runs/sd263_f16` `[file]`; `ablation_frames.log`; `fig09_temporal_ablation.png`.

### R4. How much is ImageNet initialisation worth?

* **Method:** INCLUDE-50 take-group, 120 epochs, DeiT-Tiny vs random init.
* **Result:** **29.6 % vs 23.3 %** (+6.3).
* **Verdict:** transfer helps, far less than a low-data regime suggests; the limit is in-domain data.
* **Evidence:** `runs/include50__take-group` (0.2956) vs `runs/include50_take-group_scratch` (0.2327) `[file]`.

### R5. Recipe fixes and source-cache quality

* **Result:** EMA fix (the 0.999 decay bug), 320 epochs and stronger appearance augmentation: 29.6 → **42.1 %** (+12.5). Switching the *source cache* from 80 px to 128 px with the **same 64 px model input**: 42.1 → **46.5 %** (+4.4), because area-downsampling 128 → 64 is cleaner than 80 → 64 and scale jitter gets range.
* **Verdict:** data quality beat any architectural change tested.
* **Evidence:** `runs/include50__take-group` → `runs/include50_v2__take-group` (0.4214) → `runs/abl64__take-group` (0.4654) `[file]`.

### R6. The improvement ladder on INCLUDE-263 session-disjoint (and a withdrawn claim)

All on the same 1,010 test clips; each row is added to the previous one. `run_iter3.sh`, `run_iter3b.sh`, `run_iter3c.sh`, `run_iter3d.sh`, `run_iter3e.sh`.

| # | Change | Top-1 | Δ | Survives the 2.7-point noise floor? | Run `[file]` |
|---|---|---|---|---|---|
| 0 | baseline: ImageNet init, 1,957 clips, best-validation checkpoint | 22.1 % | – | – | `sd_full263` |
| 1 | final-EMA checkpoint instead of best-validation | 25.3 % | +3.2 | no (1.2 σ) | `sd263_lastsel` |
| 2 | validation folded into train (2,845 clips) | 33.0 % | **+7.7** | **yes** (2.9 σ) | `sd263_foldval` |
| 3 | + CISLR self-supervised init (pretrained on all ~6,100 usable CISLR clips, no labels: 12× the data of the 493-clip labelled merge; `run_lever2.sh`, `lever2.log`) | 34.9 % | +1.9 | no (0.7 σ; p = 0.36) | `sd263_ssl_foldval` |
| 4 | + 6-view TTA | 36.8 % | +2.5 | yes (see §4.2) | its `summary.json` key `test_tta` |

Five seeds of row 3's configuration (`sd263_ssl_foldval{,_s1,_s2,_s3,_s4}`): **35.0 ± 1.9 %** top-1, balanced 38.4 ± 1.5, TTA 37.5 ± 2.2 (recomputed for this document).

* **A claim that was withdrawn.** Measured against the weak baseline (row 0), CISLR pretraining looked like +7.6 points at p < 0.0001 (`sd263_ssl` 29.7 %), and an earlier report credited it with the whole supervised gain. Once checkpoint selection and training-set size were fixed it was +1.9 at p = 0.36. The pretraining had been compensating for a weak recipe. The labelled-CISLR merge fared no better (p = 0.45; `sd263_cislr` 23.2 %, `sd263_ssl_cislr` 31.1 %).
* **Why selecting on validation was wrong:** validation shares every session with train (blocks {1,2,3,4,7 to 12} against test {0,5,6}), so it measures in-session recognition. SSL pretraining moved validation by −0.8 and test by +7.6.
* **Evidence:** the runs above `[file]`; `iter3*.log` `[log]`.

### R7. CISLR pretraining, corrected: a resolution mismatch (a second reversal)

* **Hypothesis:** the checkpoint tested in R6 was compromised. CISLR crops reach 128 px by *upscaling* ~78 source px; INCLUDE's by *downscaling* ~173 px. Variance of Laplacian over 300 clips per corpus: INCLUDE 58.5, CISLR 27.8, so CISLR is **2.1× blurrier** (`[report]`, technical report §11.5; no script committed).
* **Method:** `resolution_jitter` (`dataset.py:420`) degrades a clip to a random lower resolution and back during *fine-tuning only*, so INCLUDE clips are drawn from the sharpness distribution CISLR produced. `run_cislr_fix.sh`.
* **Result (two seeds):** jitter + ImageNet init (control) **30.4 %** (31.0, 29.9); jitter + CISLR SSL init **39.9 %** (39.9, 39.8), sd 0.1. **+9.5 points**, the tightest replication in the project. With TTA ≈ 42.8 %.
* **A hypothesis tested and rejected:** also applying jitter during *pretraining* (`cislr_mim_rj`). Four seeds (43.2, 36.8, 34.7, 34.1) mean **37.2, sd 4.2**: below the simpler configuration and unpredictable. Likely reason: masked reconstruction is already hard, and randomising the target's resolution makes the target ambiguous in a way classification (which can fall back on coarse shape) does not suffer. The finetune-only checkpoint is kept. `run_cislr_rj_pretrain.sh`, `run_rj2_extra_seeds.sh`.
* **Evidence:** `runs/sd263_rj_s{0,1}`, `runs/sd263_ssl_rj_s{0,1}`, `runs/sd263_ssl_rj2_s{0,1,2,3}` `[file]`; `cislr_fix.log`, `cislr_rj_pretrain.log`, `rj2_extra.log` `[log]`.

### R8. Scaling the same recipe to iSign

* **Method:** `run_isign_pipeline.sh`: extract a diverse subset of iSign (`isign_extract.py`), crop it, pretrain (`isign_mim`, 17,896 clips, 100 epochs), fine-tune with finetune-only jitter.
* **Result:** **41.8 / 42.4 % → 42.1 %**, ~44 % with TTA (43.7 / 44.3). The project's best INCLUDE-263 figure at the time, and the initialisation every later model uses.
* **Evidence:** `runs/sd263_isign_rj_s{0,1}` `[file]`; `isign_pipeline.log`, `lever2.log` `[log]`.

### R9. Cross-corpus: the headline does not generalise (the binding limitation)

* **Question:** do INCLUDE-trained models recognise ISL from a different corpus?
* **Method:** score every leak-free checkpoint on T-A (INCLUDE session-disjoint, n = 1,010) and T-B (CISLR cross-corpus, n = 609). `run_crosscorpus_eval.sh`.
* **Result:** T-B **0.3 to 1.8 % (chance 0.38 %)** for every model, including those reaching 42 % on T-A:

| Model | T-A | T-B |
|---|---|---|
| `sd263_isign_rj_s0` / `_s1` | 41.8 / 42.4 | 1.3 / 0.3 |
| `sd263_rj_s0` / `_s1` | 31.0 / 29.9 | 1.5 / 1.3 |
| `sd263_foldval` | 33.0 | 1.8 |

* **Not a label artefact:** retrieval in the model's feature space finds same-word INCLUDE clips 10× better than chance for CISLR clips, and 1-NN on the features (1.6 %) scores the same as the classifier head (1.5 %), locating the failure in the representation `[report]`.
* **Four fixes tried:** the labelled merge (none, p = 0.45); iSign pretraining (none, 1.3 %); supervised CISLR exposure of 304 clips (top-1 flat, top-5 4.3 → 9.5 %; `runs/mix50_isign_s{0,1}`, `run_multicorpus.sh`, `multicorpus.log`); freezing the encoder (`run_freeze.sh`, `freeze.log`):

| `backbone_lr_scale` | T-A | T-B |
|---|---|---|
| 0.0 (frozen) | 31.8 % | 1.0 % |
| 0.01 | 38.7 % | 0.7 % |
| 0.1 (default) | 41.8 % | 1.3 % |

  T-A rises with encoder training; T-B is flat at ~1 %. Fine-tuning was never erasing transferable features; they were never there. Masked reconstruction rewards modelling pixel appearance (blur, colour, camera response), the corpus-specific information a transferable representation should discard.
* **Later correction (R15):** CISLR failure is partly detection: MediaPipe finds a left hand in **25.4 %** of CISLR frames against 91.1 % in INCLUDE (`runs/gan_study/detection/cislr_*.json` `[file]`, §9.6). Sharpness was real but never the whole story.
* **Safe failure:** not one of 609 off-domain clips exceeded confidence 0.4 (mean confidence 0.077 against 0.304 in-domain) `[report]`.
* **Verdict:** every accuracy in this project means "on INCLUDE-like video". No measurement here supports a claim of general ISL recognition.
* **Evidence:** `crosscorpus_eval.log`, `freeze.log`, `multicorpus.log` `[log]`; `runs/frz00_isign`, `runs/frz001_isign`, `runs/mix50_isign_s*` `[file]`.

### R10. Vocabulary size, and a correction to its own conclusion

* **Method:** `build_vocab_tier.py` builds tiers ranked by *total* clips (ranking by training clips is wrong: it leaves the test set empty); `run_vocab_sweep.sh`, 250 epochs, two seeds.
* **Result:** 

| Words | Test clips | Top-1 | Top-1 + TTA |
|---|---|---|---|
| 262 | 1,010 | 42.1 ± 0.4 | 44.0 |
| 137 | 976 | 45.8 ± 0.6 | 51.0 |
| 100 | 865 | 49.6 ± 0.8 | 53.2 |
| 50 | 472 | 51.7 ± 3.3 | 56.2 |
| 30 | 278 | 41.2 ± 4.3 | 45.9 |

* **Correction:** each row is scored on a *different* test set, so "50 words is an optimum" does not follow. On the **same 472 clips**: a 50-word specialist **73.3 %** vs the 262-word model masked to 50 **75.6 %**. Training narrow is worse than training wide and deploying narrow. This is why the product is scoped to 50 words but the training set is not.
* **Evidence:** `runs/vocab{30,50,100,137}_s{0,1}` `[file]`; `vocab_sweep.log` `[log]`; the 73.3 vs 75.6 pair is `[report]` (technical report §11.7.1).

### R11. Training duration, and a negative result on grokking

* **Question:** train accuracy was 100 % with test at 41.8 %, loss at the label-smoothing floor (0.879 for 262 classes): the textbook precondition for grokking.
* **Method:** `run_grokking.sh`, `run_longtrain.sh`; 50-word tier, iSign init; weight decay 0.05 and 0.5.
* **Result:**

| Run | Epochs | Weight decay | Top-1 | + TTA |
|---|---|---|---|---|
| `vocab50_s{0,1}` | 250 | 0.05 | 51.7 | 56.2 |
| `long750_s0` | 750 | 0.05 | 55.7 | 57.4 |
| `long2000_s1` | 2,000 | 0.05 | 57.4 | 62.1 |
| `grok_wd005` | 2,000 | 0.05 | 61.9 | 64.6 |
| `long2000_wd5` | 2,000 | 0.5 | 58.7 | 60.0 |

* **Verdict:** **no grokking** (both arms climb gradually; the 10× weight-decay arm gave no transition and no gain). What the probe found instead: **every earlier run was undertrained**; 250 → 2,000 epochs is **+7.9** (51.7 → 59.6, mean of two seeds), a steady climb of ~+4 per 3× compute. Doubling again (R15) buys nothing at 16 frames, so the long-training effect was partly a symptom of the frame bottleneck.
* **Evidence:** the runs above `[file]`; `grokking.log`, `longtrain.log` `[log]`.

### R12. Reopening the capacity question at the proper budget

* **Question:** "data-limited, not capacity-limited" was concluded from 250-epoch runs. Is it still true at 2,000 epochs?
* **Method:** `run_aug_capacity.sh`, `run_frames_confirm.sh`. The iSign SSL checkpoint's time and position embeddings are sized for 8 frames × 64 px, so `train.py:75 resize_position_embeddings` was written first; otherwise every cell silently measures ImageNet init.
* **Result (50-word tier, 2,000 epochs):**

| Frames | Pixels | Seeds | Top-1 | + TTA | Top-5 |
|---|---|---|---|---|---|
| 8 | 64 | 2 | 59.7 | 63.4 | 81.8 |
| **16** | **64** | **2** | **66.1** | **67.3** | **87.1** |
| 8 | 96 | 1 | 60.4 | 62.3 | 82.4 |
| 16 | 96 | 1 | 61.0 | 61.4 | 81.1 |

* **Verdict:** doubling frames is **+6.4**, the largest single move in the project, replicated (66.7 / 65.5, spread 1.2 against the 8-frame baseline's 4.5). Resolution does nothing and combining the two undoes the frame gain. The conclusion was half right, and the wrong half was load-bearing.
* **Evidence:** `runs/{cap_8f96,cap_16f64,cap_16f64_s1,cap_16f96,grok_wd005,long2000_s1}` `[file]`; `runs/gate_16f64.json`, `gate_cap16.json`, `gate_2000ep.json` `[file]`.

### R13. A strong-augmentation bundle (a near-miss)

* **Method:** cutmix, random erasing, speed (tempo) jitter and DeiT-strength always-on mixup, tested as one bundle (decomposing something that might do nothing is the wrong order to spend 2.5 h per cell).
* **Result:**

| Frames | Baseline | Strong | Δ |
|---|---|---|---|
| 8 | 59.7 | 60.3 | +0.6 |
| 16 | 66.1 | 67.9 | +1.8 (0.7 σ) |

  Top-5 improved at both (+4.0, +1.6), but **gating got worse**: at τ = 0.4 the augmented ensemble answered 85.0 % of clips at 75.6 % accuracy; the baseline 75.6 % of clips at 80.7 %. Soft labels flatten confidence, which is a worse ranking signal for gating.
* **Verdict:** top-1 null, calibration worse; not adopted. On the first seed the 16-frame arm read 69.1 vs 66.7 and the clean story (augmentation pays once frames are not binding) would have been wrong; the second seed halved it.
* **Evidence:** `runs/aug50_s{0,1}`, `runs/aug16_s{0,1}` `[file]`; `runs/gate_aug16.json` vs `gate_cap16.json` `[file]`.

### R14. Past the 16-frame cache ceiling

* **Method:** re-extract INCLUDE at **32 frames** (`cache128_f32`, 98 min, 18.7 GB, 0 failures), confirm detection quality is unchanged (left 91.1 vs 91.3 %, right 85.2 vs 85.4 %), rebuild a leak-free tier (571 INCLUDE clips, no CISLR: `vocab50clean__session-disjoint.csv`), and read every cell from the *same 32-frame cache* varying only `n_frames`. `run_frames32.sh`, `run_jitter_headroom.sh`.
* **Result (`[file]`: `runs/f{8,16,24,32}_clean_s*`):**

| Frames drawn | Plain top-1 | + TTA | TTA gain |
|---|---|---|---|
| 8 | 60.0 | 63.8 | +3.8 |
| 16 | 69.9 / 70.3 | 73.3 / 72.0 | +3.4 / +1.7 |
| 24 | 69.7 | 70.3 | +0.6 |
| 32 | 72.5 | 72.7 | +0.2 |
| 16 of a 16-frame cache (R12) | 66.7 | 67.8 | +1.1 |

* **Verdicts:** 16 frames from a 32-frame cache beats 16 from a 16-frame cache by **+3.2 plain / +5.5 TTA** on 12 % less data (the sampler simply has somewhere to jitter). A recorded prediction that 32 frames would underperform was **wrong** (it topped plain accuracy); the TTA gain does track headroom perfectly (+3.8, +3.4, +0.6, +0.2), while plain accuracy tracks frame count. Both are real; they are separate effects. 16 frames is the deployable choice (half the inference cost of 32, statistically tied).
* **Ensembles (frame count as a diversity axis):** 2 seeds at 16 frames **74.4 %** (`runs/gate_f16.json` `[file]`); + the 32-frame model **75.0 %** (`runs/gate_mixed.json` `[file]`); a 4-member 75.4 % and the 5-member **75.8 % top-1 / 92.8 % top-5** (90.7 % accuracy while answering 75 % of clips) are `[report]` only, with no committed result file.
* **Evidence:** no root log was kept for `run_frames32.sh` or `run_jitter_headroom.sh`; the run summaries `runs/f{8,16,24,32}_clean_s*/summary.json` carry `test` and `test_tta` `[file]`.

### R15. Three levers that failed (and why they stay in the record)

Against a control of `f16_clean_s0/s1` (69.9 / 70.3, spread 0.4), one variable each, `run_push75.sh`:

| Lever | Top-1 | Δ vs control | Reason |
|---|---|---|---|
| SSL pretrained **natively at 16 frames** (`f16_mim16_s0`) | 60.0 | **−10.2** | `cache_isign` held 16 frames, so pretraining at 16 sampled 16 of 16 and had no temporal jitter. It reached a *lower* reconstruction loss (0.322 vs 0.351) and transferred *worse*: loss was reporting progress on an easy shortcut |
| +76 labelled CISLR clips (`f16_full_s0`, 13 % more data) | 64.4 | **−5.7** | MediaPipe finds a left hand in 25.4 % of CISLR frames; three quarters of CISLR left-hand crops are interpolated stale boxes. More data was worse data |
| 4,000 epochs (`f16_long4k_s0`) | 70.8 | +0.6 (null, floor 2.7) | duration has saturated at 16 frames |

Two predictions recorded in advance in this experiment were wrong (32 frames would underperform; native 16-frame pretraining would pay). Each mechanism was real but incomplete. The enforced consequence: every claim gets a second seed and a matched control before it is written down.

### R16. Meeting the 2 MB budget

The 75.8 % headline had been a five-model ensemble at 18.5 MB, and the best single model weighed 14.55 MB. Compression came before more accuracy work ("a number the product cannot ship is not a result"). Pixel-only ladder in §7.4; method and bugs in §7.4. `run_lm_qat.sh` re-ran QAT in the corrected form (fixed 300 epochs, final weights) for the landmark model.

* **A control that turned out not to be needed:** on the 262-way single-view metric QAT finished 2.2 points *above* its FP32 start, which QAT has no business delivering. The suspected cause (300 extra epochs) was to be separated by an FP32 control; it was not run because on the deployed metric QAT and FP32 both score 75.6 % (the same 357 of 472 clips). The claim therefore narrows to what needs no control: **QAT recovers the 1.2 points post-training quantisation loses, and nothing more.**
* **Evidence:** `runs/f16_262w_s{0,1,2}`, `runs/f16_262w_s0_qat4` `[file]`; `runs/f16_262w_lm_s*_qat4` `[file]`; `lm_qat.log` `[log]`.

### R17. Fine-tuning on the deployed objective (rejected)

* **Method:** `islvit/narrow.py`, `run_narrow.sh`: train the 262-word model for a fixed 150 epochs with the softmax restricted to the 50 deployed columns.
* **Result:** seed 0 75.6 → 78.4 (20 clips gained / 7 lost, p = 0.019); seed 1 73.5 → 73.1 (12/14, p = 0.85); seed 2 72.0 → 68.4 (11/28, p = 0.009). Mean 73.7 → 73.3, p = 0.60.
* **Verdict:** two seeds moved significantly in *opposite* directions: an unstable procedure, not a small effect. Seed 0 alone would have read as a significant +2.8.
* **Evidence:** `runs/f16_262w_s{0,1,2}_narrow/summary.json` `[file]` (`masked50_before`, `masked50`); `narrow.log` `[log]`.

### R18. The hand-landmark stream (the largest single gain)

* **Idea:** the crop pipeline already ran MediaPipe, used the 21 joints per hand to draw a box, and discarded them. Handshape, measured directly, was being thrown away on every clip.
* **Method:** `islvit/data/landmarks.py` plus `--landmarks` (§5.3, §6.3), changing nothing else of the `f16_262w` recipe. `run_landmarks.sh`.
* **Result:** pixel-only → landmark on the clean test cache: **75.0 / 73.7 / 72.2 (mean 73.7) → 87.3 / 86.9 / 85.4 (mean 86.5)**; paired gained/lost 76/18, 83/21, 88/26, p = 1.2e-9, 6.7e-10, 4.6e-9. On all 262 words: 61.7 / 66.7 % against ~52 % pixel-only. INT4 QAT then gives **84.5 / 87.5 / 85.8 = 85.9 %**.
* **Why it is believable:** landmarks cannot carry the label (each is computed from that clip's own frames), and blanking inputs at test time shows complementarity: landmarks alone reach ~70 %, landmarks plus pixels ~87 %. The shared-signer caveat applies to joints as it does to faces.
* **Verdict:** this is the largest gain in the project, from an input already computed; every capacity, resolution and schedule lever in R12 to R15 was worth a few points.
* **Seed 2:** crashed at epoch ~235, resumed from its checkpoint, scored on its own; its paired gain (88/26) replicates the effect (`run_lm_s2_resume.sh`, `lm_s2_resume.log`).
* **Evidence:** `runs/f16_262w_s{0,1,2}`, `runs/f16_262w_lm_s{0,1,2}`, `..._qat4` `[file]`; `landmarks.log`, `landmarks_train.log` `[log]`; `docs/figures/fig19_walkthrough.png` (a real clip traced stage by stage by `islvit/walkthrough.py`).

### R19. The detector turned out to be stateful (§5.5)

Found while verifying that live inference reproduces offline numbers. Fixed by `reset_detectors`; test set re-extracted; every model re-scored; nothing moved by more than 0.6 points. Evidence in §5.5.

### R20. Shipping (§7.5)

`release/*.pt`, loader, CPU path, verified identical scores (84.5 / 87.5 / 85.8). Commit `a5fef61`.

### R21. Validation screening of landmark levers: none survives

* **Why a new protocol:** pushing past 85.9 % means trying many ideas, and choosing between them on the 472 test clips would slowly fit the test set. Screening therefore used the *original* session-disjoint split: train 1,957 clips, validation 888 (154 deployed-word clips, all 50 words present), test untouched; the winner would be retrained on train + validation and scored on test once (`mask50 --eval-file/--eval-split`, commit `d2b95f0`).
* **Result (masked-50 TTA on the 154 validation clips, 500 epochs):**

| Variant | Top-1 | Verdict |
|---|---|---|
| base, seed 0 / seed 1 | 83.1 / 86.4 | – (the same recipe varies ~8 points between seeds on 154 clips) |
| `--lm-aug 1.0`, seed 0 / seed 1 | 86.4 / 77.9 | **does not replicate** (pooled +10 / −18); rejected |
| `--lm-aug 2.0` | 80.5 | worse |
| `--lm-interp` | 82.5 | null |
| `--lm-velocity` | 78.6 | likely harmful (+4 / −11) |
| `--lm-pair` (two-hand relation) | 77.9 | worse |
| `--lm-wrist-vel` (wrist trajectory) | 84.4 | null (+6 / −4) |
| 1,000 epochs | 83.1 | null |

* **The error analysis that motivated the levers:** wrong clips had the same hand coverage as right ones (~90 %), so detection was not the bottleneck; ~70 % of errors fell in six confusion pairs (you / you-plural, big / wide / narrow, short / small, good / bad / wet, Dress / T-Shirt), distinguished by wrist trajectory or two-hand relation, neither an explicit input. The levers addressed that and still did not pay.
* **Evidence:** `runs/dev_lm_*/val50.json` `[file]` (key `masked_tta.top1`, n = 154); `dev_screen.log`, `dev_aug.log`, `dev_pair.log` `[log]`; commit messages `42080f0`, `8d0bc41`; the confusion groups are hard-coded in `islvit/ui/src/core.js` (`DATA.lookalikes`) and drive the Enrolment queue.

### R22. The ensemble ceiling

* **Method:** `islvit/ensemble_val.py` averages the ten train-only validation models.
* **Result:** single models 82.1 % mean; ensembles of 3 reach 84.1 %; all ten 82.5 %. Averaging is worth ~2 points, so distilling an ensemble would recover perhaps 1. The remaining errors are look-alike pairs that ~16 clips per word do not separate.
* **Evidence:** commit `8d0bc41` message `[report]`; cached probabilities `runs/_val_probs/*.npz` `[local]`.

### R23. Error structure of the model

* Performance is **bimodal**: on INCLUDE-263 session-disjoint 42 % of words sit at 0 % recall and 9 % at 100 % (INCLUDE-50: 60 % and 27 %); the model learned a subset of the vocabulary and has essentially no signal on the rest.
* Top-5 is far higher than top-1 (96.4 % vs 85.9 % for the shipped model), which is why the Not-sure screen offers five candidates.
* Most frequent confusions are listed in technical report Tables 14.2 and 14.3 `[report]`; per-word recall of the shipped models on the 472 clips is embedded in the interface (`islvit/ui/src/core.js` `DATA.recall`, sourced from `runs/f16_262w_lm_s*_qat4/masked50_clean.json`) and feeds the Vocabulary screen. `docs/figures/fig07_per_class_include50.png`, `fig08_per_class_full263.png`, `fig18_per_class_50.png` plot it.

---

## 9. The super-resolution GAN study

Full write-up: `docs/GAN_COMPARATIVE_STUDY.md` (+ `.docx`, `.pdf`), generated by `python -m islvit.gan_report`, which reads every number from `runs/` when it runs. The plan that led to version 3 is `docs/GAN_IMPROVEMENT_PLAN.md`. This section summarises the design and results and points to the files.

### 9.1 Question, placement and rejected alternatives

**Question:** when input video is low-resolution, does a super-resolution GAN in front of the recogniser give better accuracy than not using one? "Not using one" has two meanings, and both are tested: **bicubic upscaling** (what the pipeline did) and the **same generator trained without the adversarial loss** (SR-L1, and in v3 the hand-loss arm). Without the second control, a GAN that beats bicubic would prove only that learning to upscale helps.

**Placement.** Every clip reaches the hand detector through one function, `crops.py::read_sampled_frames`, which resizes video below 720p up to 720p with bicubic interpolation. That line is where the project's low-resolution failures pass (CISLR's 300 px frames are upscaled 2.4×; a phone or glasses camera would be too). The generator replaces only that resize (§5.1). Rejected alternatives:

* *Synthesising training clips with a GAN* (about 15 clips per word cannot train a video generator that keeps signs correct, and label noise on confusable words is a risk).
* *Upscaling the crops after detection* (cannot help the detector, which is where low resolution does most damage).

### 9.2 The arms and where each lives

| Arm | GAN? | Generator | Training | Run dir (`summary.json`, `history.csv` `[file]`; `generator.pt` `[local]`) |
|---|---|---|---|---|
| Bicubic | no | none | – | – |
| **SR-L1** | no | SRVGG-compact 64 ch × 16 layers, 621,424 params | pixel L1, 100,000 iterations, batch 32, LR 2e-4, 55.6 min | `runs/sr_l1` |
| **SR-GAN v1** | yes | same | from SR-L1: L1 + VGG19 perceptual + adversarial (BCE), 60,000 iterations, batch 16, LR 1e-4, 236.8 min | `runs/sr_gan_v1` |
| **SR-GAN v2** | yes | same | from v1 with a fresh discriminator, 80,000 more iterations at LR 5e-5, 261.4 min | `runs/sr_gan` |
| v3 base (L1) | no | 96 ch × 24 layers, 2,039,568 params | pixel L1, 100,000 iterations, batch 32, new data and degradations, 140.3 min | `runs/sr3_l1` |
| **SR-GAN v3** | yes | v3 | from v3 base: L1 + VGG19 + **RaGAN** + hand-landmark losses, 60,000 iterations, batch 16, LR 1e-4, 280.2 min | `runs/sr3_gan_task` |
| v3 GAN, no hand losses | yes | v3 | as v3 without the hand losses, 259.3 min (validation only) | `runs/sr3_gan` |
| **v3 without the GAN** | no | v3 | from v3 base: L1 + hand losses, no discriminator, LR 1e-4, 60,000 iterations, 74.5 min | `runs/sr3_l1_task` |

Each GAN has a matching no-GAN control from the same generator and data: SR-L1 for v1 and v2, "v3 without the GAN" for v3. (Note: in the code and in the old repository docs, v2 is the directory `runs/sr_gan` and `ft360_srgan_*`; v1 is `runs/sr_gan_v1` and `ft360_srgan_v1_*`; the v1 study results are archived under `runs/gan_study_v1/`.)

### 9.3 Networks and losses

Code `islvit/sr/models.py`; follows Real-ESRGAN (Wang et al., ICCV-W 2021) at a size this project can afford.

* **Generator** (`SRGenerator:32`): Real-ESRGAN's compact VGG-style network. 3×3 convolutions with PReLU at the *low* resolution, a final ×4 pixel shuffle, and the output is a **residual over a nearest-neighbour upsample** of the input, so an untrained or unsure network falls back to plain upscaling instead of noise. All work is at low resolution, so full frames are cheap.
* **Discriminator** (`UNetDiscriminatorSN:51`): a U-Net with spectral normalisation, 4,376,897 parameters, returning a realness score **per pixel**, so it can penalise an implausible fingertip without penalising the palm beside it. Exists only during training.
* **Perceptual loss** (`VGGPerceptualLoss:84`): L1 between VGG19 features before the ReLUs at conv1_2, conv2_2, conv3_4, conv4_4, conv5_4, weights 0.1, 0.1, 1, 1, 1. The code falls back to ResNet-50 if VGG weights cannot be fetched; **every GAN run's `summary.json` records `"perceptual": "vgg19"`**, so the fallback was never used.
* **Total generator loss** (`sr/train.py:283-311`): `1.0·L1 + 1.0·perceptual + 0.1·adversarial` (+ `2.0·landmark-error + 0.2·feature-distance` in v3).
* **Adversarial form:** v1 and v2 use binary cross-entropy against all-ones; v3 uses the **relativistic average** loss (real and fake scores compared against each other's mean), which is less prone to a runaway discriminator.
* **Optimisation:** Adam (β = 0.9, 0.99), cosine decay to 1 % over the fixed schedule, generator **EMA 0.999** and the **final EMA weights are what ships**; the discriminator and VGG run in bfloat16, the generator stays float32 (in float32 a GAN step ran at 1.4 it/s, so the planned 100,000 adversarial iterations were cut to 60,000 *before* the run; bfloat16 and batch 16 gave 4.2 to 4.3 it/s). Resume checkpoints every 2,000 iterations with a fingerprint check.
* **No selection by results:** iteration counts and weights are fixed in advance; the `selection` field of every `summary.json` reads "final EMA weights, iteration count fixed in advance".

### 9.4 Training data and degradations

| Bank `[local]` in `C:/isl_gan_work/` | Contents | Built from | Used by |
|---|---|---|---|
| `hr_train.npy` (+`.json`) | 32,000 patches of 128×128: 16,000 hand/face crops (genuine detections only) + 16,000 random patches of whole 720p frames, from **2,845 clips** | `splits/full263__session-disjoint-foldval.csv` (**includes the validation sessions**) | v1, v2 |
| `hr_train_v3.npy` | same counts from **1,957 clips** | `splits/full263__session-disjoint.csv`, train rows only (**validation sessions excluded**) | v3 |
| `hands_train_v3.npy`, `hp_train.npz` | 43,682 MediaPipe-verified hand crops | original train sessions | v3 (half of every batch) and the hand proxy |
| `hp_val.npz` | hand crops from the validation sessions | | proxy validation; v3 development |
| `hr_heldout.npy` | 600 crop + 1,200 frame patches from 1,412 held-out-session clips (test + unused) | | image quality only (1,200 patches used) |

* **Leak guard** (`sr/patches.py`): the training bank draws only from `train` rows and asserts none is a test, unused, or 472-clip video.
* **Random degradation** (`sr/degrade.py:122 random_degrade`), a simplified Real-ESRGAN second-order pipeline: *first order*: blur (90 %; 25 % directional motion blur, otherwise Gaussian σ 0.2 to 2.4), random resize ×0.4 to 1.3 with a random interpolation, noise (Gaussian, 40 % of those colourless, or Poisson shot noise), JPEG quality 40 to 95; *second order* (80 %): milder repeat; then a final ×4 downscale with compression either before or after (a camera scales then compresses; a re-shared clip is scaled after compression). **v3 adds H.264 intra coding** (`h264:101`, CRF drawn from 23, 28, 33, 38, 43) on 50 % of samples, because real phone and glasses video is H.264/H.265, not JPEG; a generator that never saw a codec cannot undo one. Persistent per-(size, CRF) encoders cost 3 to 6 ms instead of ~1 s.
* **Deterministic study degradation** (`degrade_frame:35`): area downscale to the target height, then one OpenCV JPEG encode, so every arm sees byte-identical inputs (§9.7).

### 9.5 The three versions: what, how they differ, why each was built

| | **v1** | **v2** | **v3** |
|---|---|---|---|
| Why built | The planned design, built as Real-ESRGAN describes and compared with bicubic and SR-L1 (pre-registered) | v1's discriminator had won (scored real 0.86, fake 0.14; generator adversarial loss 3.00) and v1 ≈ SR-L1: was v1 undertrained? **Not pre-registered** | Diagnostics (§9.8) put ~74 % of the remaining 180p loss on the hand-landmark side, which a texture-realistic GAN does not address. **Not pre-registered** |
| Generator | 621,424 params | same | 2,039,568 (3.3×) |
| Starts from | SR-L1 | v1 + new discriminator | v3 L1 base |
| Adversarial loss | BCE | BCE | RaGAN |
| Other losses | L1, VGG19 | L1, VGG19 | L1, VGG19, **hand-landmark error (2.0), hand features (0.2)** |
| Degradation | blur/resize/noise/JPEG | same | + H.264 on half |
| Data | train + validation sessions | same | train sessions only + 43,682 hand crops (half of each batch) |
| Adversarial iterations | 60,000 | 140,000 in total | 60,000 |
| Discriminator at the end, real / fake | 0.86 / 0.14 (adv 3.00) | 0.79 / 0.21 (adv 2.24) | 0.75 / 0.17 |

Evidence: `runs/{sr_gan_v1,sr_gan,sr3_gan_task}/history.csv` columns `d_real`, `d_fake`, `adv`; the table is built from the last row (`islvit/gan_report.py last_history`).

**The hand proxy** (`islvit/sr/handproxy.py`, `runs/hand_proxy/summary.json` `[file]`). MediaPipe is not differentiable, so a generator cannot be trained against it directly. The proxy is a **ResNet-18** trunk whose features are expanded by three transposed convolutions into 32×32 heatmaps for the 21 joints, read out by **soft-argmax**, so its output stays differentiable. (A flatten-and-regress head was tried first and underfit at 0.094 mean error.)
* *Data:* 43,682 hand crops whose box is MediaPipe's own (direct detection), landmarks recomputed with `crops.box_from_points`, kept only if they match their stored geometry within `GEOMETRY_TOLERANCE` 0.0012 (an early label misalignment came from a crop cache and a landmark cache built from different detections). Original train sessions only; validation sessions held out; the 472 test clips never touched (asserted in `split_rows`).
* *Training:* 20 epochs, final weights, geometric augmentation only (flip, ±15° rotation, ±10 % scale; no blur or noise, so a degraded-looking hand *reads differently* and that difference is the signal).
* *Quality on 2,466 validation crops:* mean landmark error **0.0215** of the crop width, median 0.0116, PCK@0.05 **91.4 %**, PCK@0.10 97.4 %. For reference MediaPipe re-run on the same crops differs from its own stored labels by 0.041 mean / 0.019 median.
* *Use:* frozen during GAN training. `hand_terms` penalises the distance between the proxy's landmarks on the generator's output and on the clean crop (weight 2.0) and between their intermediate features (weight 0.2). Hand crops are 50 % of every batch and are not rotated by 90° (the proxy reads upright hands).
* *Development metric (`ValHands`):* proxy landmark error on validation hand crops degraded like 180p video, final values `[file]` `runs/sr3_*/summary.json` key `final_val.val_hand_err`: bicubic **0.0862**; v3 base 0.0393; v3 GAN no hand losses 0.0387; **SR-GAN v3 0.0346**; v3 without the GAN **0.0264**.

### 9.6 Study protocol: what was measured and how

| Item | Setting | Code / file |
|---|---|---|
| Test clips | The 472 held-out clips of the 50 deployed words, identical in every arm | `splits/_test472_paths.csv` |
| Degradation levels | **360p / JPEG 60** (×2 upscale to 720p), **240p / 50** (×3), **180p / 40** (×4), applied inside the frame reader before detection | `run_gan_study.sh` `LEVELS`; `degrade_frame` |
| Front ends | bicubic; SR-L1; v1; v2; v3 (SR-GAN v3); v3 without the GAN | `--sr <generator.pt>` |
| Generator output | ×4, then area-resized to exactly 720p (Real-ESRGAN "outscale"), so one network serves ×2, ×2.4, ×3 and ×4 | `sr/upscaler.py:84 upscale_frames` |
| Per front end | 32 frames per clip, 128 px crops, landmarks through the same front end; scored with `islvit.mask50` | `run_gan_study.sh` `extract`, `score` |
| Models scored | **Frozen INT4:** the three release models, unchanged. **Frozen FP32:** `runs/f16_262w_lm_s{0,1,2}`. **Fine-tuned:** the FP32 models fine-tuned for **150 epochs** on the 2,845 training clips degraded to 360p/JPEG 60 through bicubic, v1 or v2 (LR 1e-4, resolution jitter 0.5, 16 frames, landmarks, final epoch, three seeds), to ask whether a GAN still helps once the classifier has adapted. v3 was **not** fine-tuned against | `runs/ft360_*` |
| Statistics | Exact McNemar pooled over seeds; pre-registered rule (§4.3) | `islvit/gan_report.py` |
| Detection and hallucination | Against the clean 1080p extraction of the same clips: genuine share per stream; **extra hands** (found in degraded but not in clean video: how a GAN that invents hands would show); **missed hands**; median wrist error; **misplaced hands** (> 5 % of frame height apart) | `islvit/sr/study.py:69 detection` |
| CISLR | Natively 300 × 300; detection only (the classifier is at chance there for reasons unrelated to resolution) | `islvit/sr/diag_cislr.py`, `runs/gan_study/detection/cislr_*.json` |
| Image quality | 1,200 held-out patches degraded to each level: PSNR, SSIM (own PyTorch implementation), sharpness (variance of the Laplacian), ms per patch on the GPU | `islvit/sr/evaluate.py`; `runs/gan_study*/sr_quality*.json` |
| v3 development | 200 validation-session clips through real MediaPipe against the clean landmarks, **before** any v3 test run; noise floor ≈ 1.5 % of hand frames when clean video is re-extracted | `islvit/sr/diag_detector.py`; `run_gan_v3.sh`, `run_gan_v3_dev.sh` |
| v3 test extraction | Generator in **fp16** (largest pixel difference 1/255, mean 0.03/255, 1.8× faster); recorded as `sr_precision: fp16` in each cache's `FRONTEND.json`. Only the two v3 arms that bracket the validation trade-off were run on test | `run_gan_v3_test.sh`, `sr/upscaler.py:31 half_precision` |

**Verification techniques used** (each guards against a failure that would silently invalidate the comparison):

* *Leak guards:* GAN training banks reject any test-session or 472-clip video; fine-tuning clips come from the training split only.
* *Clean video untouched:* crops from clean 1080p clips extracted with and without `--sr` were byte-identical `[report]`.
* *No front-end mixing:* every cache records degradation, generator path, generator SHA-256 and precision in `FRONTEND.json`; landmarks are refused if built through a different front end than the crops beside them.
* *Deterministic detection:* a fresh detector per video (§5.5).
* *Fixed iteration counts, loss weights, epochs and decision rule*; final weights always.
* *Diagnostics that separate causes:* the 2×2 decomposition, detector-lever control, and CISLR check (§9.8).
* *Resume-safe, guarded scripts* (`run_gan_study*.sh`, `run_gan_v3*.sh`): finished steps are skipped; deletion goes through the guarded `study drop`. (One run, v2 lane B, died of CUDA out-of-memory because eleven processes each held a CUDA context; `run_gan_v2_rest.sh` is the recovery script.)

### 9.7 Results

All accuracies: masked-50, 6-view TTA, mean of 3 seeds, 472 clips. Clean 1080p = **85.9 %** (no generator involved).

**Frozen INT4 models** `[file]` `runs/gan_study/frozen_int4/<level>_<arm>_s<seed>.json` (v1: `runs/gan_study_v1/frozen_int4/`), key `masked_tta.top1`:

| Input | Bicubic | SR-L1 (no GAN) | v3 w/o GAN | SR-GAN v1 | SR-GAN v2 | SR-GAN v3 |
|---|---|---|---|---|---|---|
| 360p, JPEG 60 | 81.2 | 83.5 | 83.8 | 84.3 | 83.1 | 84.0 |
| 240p, JPEG 50 | 76.0 | 80.9 | 79.2 | 81.8 | 82.3 | 81.6 |
| 180p, JPEG 40 | **61.0** | 76.0 | 75.6 | 75.4 | 75.5 | 75.4 |

**Fine-tuned on 360p video** `[file]` `runs/gan_study/retrained/` and `runs/gan_study_v1/retrained/`:

| Input | Bicubic | SR-GAN v1 | SR-GAN v2 |
|---|---|---|---|
| Clean | 85.8 | 86.4 | 85.9 |
| 360p | 83.5 | 84.6 | 84.9 |
| 240p | 77.9 | 83.6 | 83.1 |
| 180p | 64.4 | 77.8 | 76.9 |

**Paired verdicts** (frozen INT4; Δ in points; p exact McNemar pooled over seeds; the table is complete in `docs/GAN_COMPARATIVE_STUDY.md` §7.2):

| Comparison | 360p | 240p | 180p |
|---|---|---|---|
| learned upscaler vs bicubic: SR-L1 | +2.3 (p 0.0012) helps | +4.9 (3e-7) helps | +15.0 (4e-34) helps |
| learned upscaler vs bicubic: SR-GAN v2 | +1.9 (0.017) helps | +6.4 (8e-11) helps | +14.5 (7e-32) helps |
| **GAN vs its no-GAN control**: v1 vs SR-L1 | +0.8 (0.19) none | +0.9 (0.16) none | −0.6 (0.47) none |
| v2 vs SR-L1 | −0.4 (0.53) none | **+1.5 (0.022) helps** | −0.5 (0.58) none |
| v3 vs v3 without the GAN | +0.2 (0.80) none | **+2.4 (0.001) helps** | −0.2 (0.86) none |
| v2 vs v1 (does longer training help?) | **−1.2 (0.021) hurts** | +0.6 (0.34) none | +0.1 (0.91) none |
| v3 vs v2 | +0.9 (0.14) none | −0.7 (0.33) none | −0.1 (1.0) none |

Reading: a learned upscaler beats bicubic at every level, by 14 to 15 points at 180p. The GANs reached "helps" against their own no-GAN control in **2 of 9** comparisons, both at 240p. The second win is weaker than it looks: v3 without the GAN was the weakest learned front end at 240p (it is 3.1 points below v2, p = 5e-6), and SR-GAN v3 did not beat SR-L1 there. On the **32-bit** models the learned upscalers gain nothing at 360p (all "no reliable difference" vs bicubic; at ×2 the detector still finds most hands); at 240p the one GAN win is v1 over SR-L1 (+1.6, p = 0.018) and SR-GAN v3 is 1.7 points *worse* than v2 (p = 0.021). After fine-tuning, v1 and v2 still beat bicubic by 5.2 to 5.7 (240p) and 12.5 to 13.3 (180p); v2 is not better than v1.

**Detection** (share of frames with a genuine hand detection, left / right; clean INCLUDE is 91.1 / 85.2) `[file]` `runs/gan_study/detection/<level>_<arm>.json`, `runs/gan_study_v1/detection/`:

| Input | Arm | Left | Right | Face | Extra hands | Missed hands | Misplaced |
|---|---|---|---|---|---|---|---|
| 180p | Bicubic | 51.0 | 53.8 | 33.5 | 1.9 | 39.3 | 8.4 |
| 180p | SR-L1 | 68.2 | 69.4 | 98.6 | 2.6 | 23.5 | 5.1 |
| 180p | v3 w/o GAN | 65.5 | 63.7 | 96.3 | 1.0 | 26.1 | 3.5 |
| 180p | SR-GAN v1 | 67.9 | 72.0 | 99.8 | 2.6 | 22.4 | 4.4 |
| 180p | SR-GAN v2 | 68.6 | 71.9 | 99.9 | 2.6 | 22.1 | 4.7 |
| 180p | SR-GAN v3 | 77.6 | 76.9 | 99.2 | 3.1 | 15.5 | 4.6 |
| 240p | Bicubic / SR-GAN v2 / v3 | 71.1 / 83.2 / 85.3 | 71.2 / 81.3 / 84.1 | 86.9 / 99.9 / 99.9 | 2.3 / 2.7 / 3.3 | 20.8 / 10.2 / 8.3 | 5.1 / 2.8 / 2.6 |

The GANs did not buy accuracy by inventing hands (extra hands rise from 1.9 % to 2.6 to 3.1 %, so they invented a few). CISLR left-hand detection, which is natively 300 px: bicubic 25.0, SR-L1 25.9, v1 24.8, v2 24.3 (SR-GAN v3 was not run there).

**Where the remaining 180p loss comes from** (2×2 decomposition; frozen INT4) `[file]` `runs/gan_study/diagnostics/res_clean_pix_L3_*_s*.json` (clean pixels, degraded landmarks) and `.../mirror/degraded_pixels_clean_landmarks_L3_*_s*.json` (degraded pixels, clean landmarks):

| Front end | Both degraded | Pixels clean, landmarks degraded | Pixels degraded, landmarks clean | Clean both |
|---|---|---|---|---|
| Bicubic | 61.0 | 73.7 | 79.0 | 85.9 |
| SR-GAN v2 | 75.5 | 78.2 | 83.1 | 85.9 |

With SR-GAN v2, 10.4 points are missing at 180p: about **7.7 on the landmark side** and **2.8 on the pixel side** (≈ 74 % landmark). With bicubic both sides lose heavily (12.2 and 6.9) and compound (24.9, more than their sum). Perfect pixels alone would lift 75.5 only to 78.2; a perfect landmark extractor would lift it to 83.1.

**Detector-side levers do not fix it** `[file]` `runs/gan_study/diagnostics/detector_levers.json` (script `islvit/sr/diag_detector.py`, `runs/gan_study/diagnostics/run_levers.sh`): landmarks re-extracted from the *same* super-resolved 180p frames with different detector settings, each scored beside clean crops:

| Variant | Top-1 | Hand frames found (clean 89.7 %) | Extra | Misplaced |
|---|---|---|---|---|
| Study settings | 78.2 | 70.3 | 2.6 | 4.7 |
| Lower confidence thresholds | 79.0 | 82.1 | 5.1 | 7.1 |
| ROI upscale 256 → 512 px | 78.4 | 68.0 | 2.5 | 4.4 |
| Multi-scale ROI rescue | 78.2 | 84.5 | 5.3 | 10.7 |
| All three | 78.7 | 92.3 | 7.7 | 12.0 |
| Study settings + gap filling (≤ 4 frames) | 78.2 | 87.8 | 8.2 | 8.4 |
| All three + gap filling | 78.1 | 99.2 | 10.2 | 14.1 |

Coverage can be pushed past the clean figure (then it includes hands that are not there) while accuracy moves by at most +0.8: every extra detection brings more misplaced and spurious ones. **The limit is landmark *quality*, not whether a hand was found.** Landmark jitter at 180p: error-trajectory high-frequency energy 0.079 (SR-GAN) vs 0.107 (bicubic) against 0.055 for the clean wrist's own motion; lag-1 autocorrelation ≈ 0.49 (`[report]`, `docs/GAN_IMPROVEMENT_PLAN.md` §7.3; no script isolates it).

**v3 on the validation sessions** (200 clips; MediaPipe on super-resolved video vs the clean video) `[file]` `runs/gan_study/diagnostics/dev_v3_L3.json`, `dev_v3_L2.json`; clean video finds hands in 88.3 % of frames:

| Arm | Hands found 180p | Handshape error 180p (×100) | Hands found 240p | Handshape error 240p |
|---|---|---|---|---|
| Bicubic | 43.3 | 2.32 | 64.3 | 1.26 |
| SR-L1 | 64.6 | 1.30 | 78.8 | 0.83 |
| SR-GAN v2 | 63.3 | 1.30 | 78.3 | 0.84 |
| v3 base (L1) | 69.2 | 1.19 | 80.8 | 0.74 |
| v3 GAN, no hand losses | 71.7 | 1.28 | 83.2 | 0.80 |
| **SR-GAN v3** | **73.3** | 1.23 | **84.0** | 0.76 |
| v3 without the GAN | 60.7 | **0.99** | 79.8 | **0.66** |

v3 did what it was designed to do at detection level (most hands found; lowest handshape error without the GAN), and the test-set detection table agrees (77.6 / 76.9 % at 180p). **It did not become better classification.** One untested explanation: the frozen classifier learned from MediaPipe's landmarks on clean video and reacts to the detector's remaining errors in ways the proxy distance does not capture; fine-tuning on v3's output would test it.

**Image quality** on 1,200 held-out patches at 180p `[file]` `runs/gan_study/sr_quality.json` (bicubic, SR-L1, v2), `runs/gan_study_v1/sr_quality.json` (v1), `runs/gan_study/sr_quality_v3_task.json`, `sr_quality_v3_base.json` (v3):

| Arm | PSNR (dB) | SSIM | Sharpness (originals 73) | ms per patch (GPU) |
|---|---|---|---|---|
| Bicubic | 30.07 | 0.856 | 4 | – |
| SR-L1 | 31.83 | 0.900 | 16 | 1.02 |
| SR-GAN v1 | 31.07 | 0.883 | 31 | 0.81 |
| SR-GAN v2 | 31.09 | 0.884 | 32 | 0.83 |
| v3 without the GAN | 29.96 | 0.863 | **61** | 1.47 |
| SR-GAN v3 | 29.76 | 0.870 | 28 | 1.27 |

PSNR and SSIM reward smooth averages, so a non-adversarial upscaler is expected to beat a GAN on them (the perception-distortion trade-off); the GAN roughly doubles sharpness over SR-L1 and that does not become accuracy. **A side finding:** v3 without the GAN has the highest sharpness of all (above the originals at 360p) because the hand-landmark loss rewards whatever makes the proxy's joints land correctly and nothing in it objects to unnatural texture: its output is covered in a fine regular dot grid (visible in `docs/figures/fig23_gan_montage.png`). The same network trained with the discriminator shows no grid. This is the one place the adversarial loss did the job it exists for.

**Cost:** v1, v2 and SR-L1 share one 0.62 M-parameter network and cost the same at inference; v3's generator has 3.3× the parameters and was 1.5 to 2.3× slower per patch. No latency was measured on wearable hardware.

### 9.8 Conclusions and limits

* A learned upscaler in front of the hand detector recovers most of the accuracy lost on low-resolution video: up to **+15 points at 180p** on the shipped model.
* Training it as a GAN adds little to recognition: 2 of 9 GAN-vs-control tests passed; training longer (v2) did not help; a landmark-aware generator (v3) improved detection statistics but not classification.
* After super-resolution the remaining error is mostly in where the detector places the hand joints (≈ 74 %), and detector tuning does not recover it.
* These results do **not** show that a GAN cannot help this recogniser; they show the GANs built here did not, on synthetically degraded studio video and a frozen or partly adapted classifier.
* **Limitations:** synthetic degradation (downscale + one JPEG) rather than real phone/glasses video; one studio; v2 and v3 designed after earlier results were seen (judged by the same rule; v3's arms chosen on validation sessions); v3 not fine-tuned against; two v3 arms validation-only; v3 trained on less INCLUDE data than v1/v2 (validation sessions excluded); three seeds with ~2-point seed spread, so effects of a point or two cannot be settled.
* **Most direct next test:** fine-tune the classifier on SR-GAN v3's output and repeat on real low-resolution recordings.

### 9.9 Where every GAN-study file is

| What | Path |
|---|---|
| Report (MD/DOCX/PDF) | `docs/GAN_COMPARATIVE_STUDY.{md,docx,pdf}`; generator `islvit/gan_report.py`; figures `docs/figures/fig21_gan_accuracy.png`, `fig22_gan_detection.png`, `fig23_gan_montage.png` |
| Improvement plan and cheap-check results | `docs/GAN_IMPROVEMENT_PLAN.md` |
| Accuracy, frozen INT4 / FP32 / fine-tuned | `runs/gan_study/{frozen_int4,frozen_fp32,retrained}/`; v1 in `runs/gan_study_v1/` |
| Detection and hallucination | `runs/gan_study/detection/`, `runs/gan_study_v1/detection/` |
| Image quality | `runs/gan_study/sr_quality*.json`, `runs/gan_study_v1/sr_quality.json` |
| Diagnostics | `runs/gan_study/diagnostics/` (`res_clean_pix_*`, `mirror/`, `detector_levers.json`, `dev_v3_L{2,3}.json`, `v3_test_summary.json`, `run_*.sh`) |
| Training records | `runs/{sr_l1,sr_gan_v1,sr_gan,sr3_l1,sr3_gan,sr3_gan_task,sr3_l1_task,hand_proxy}/` |
| Fine-tuned classifiers | `runs/ft360_{bicubic,srgan_v1,srgan}_s{0,1,2}/` (note: their own `summary.json` `test` reads `top1 0.0, n 0` because the 360p training cache holds only the 2,845 training clips, so all 1,010 test clips are "absent from the crop cache, skipped" (`gan_study.log`); the real scores are in `runs/gan_study/retrained/`) |
| Drivers | `run_gan_study.sh` (bicubic, SR-L1, v1), `run_gan_study_v2.sh` + `run_gan_v2_rest.sh` (v2), `run_gan_v3.sh` + `run_gan_v3_dev.sh` + `run_gan_v3_test.sh` (v3) |
| Logs | `gan_study.log`, `gan_v2*.log`, `gan_v3*.log`, `hand_proxy.log`, `patches_v3.log` `[local]` (untracked) |
| Working data | `C:/isl_gan_work/` `[local]` (§2.4) |

---

## 10. The application: server, interface, demo

### 10.1 Architecture

```
Browser (one self-contained page)                 Python server (standard library only)
  Capture / Not sure / Sentence / Settings /  ──► GET  /          the page
  Detection / Vocabulary / Enrolment              GET  /info      model name, word list, frame/crop sizes, params
                                                  POST /predict   raw video bytes  → classify()
                                                  POST /correct   JSON             → appended to runs/corrections.jsonl
                                                  POST /enrol     raw video + X-Signer, X-Word headers → custom/<signer>/<word>/takeN
```

* **`islvit/serve.py` (1,023 lines, owned by the user and left unmodified by the Review 3 work).** A `ThreadingHTTPServer` with no framework dependency. Uploads arrive as the raw request body, not multipart (removes a class of encoding bug). `classify:50` runs the *same* `predict.extract` / `predict.extract_landmarks` / `predict.views` as the offline pipeline, averages the softmax over the views, and returns `frames`, `views`, `detection_rate` (share of hand-frames with a genuine detection), per-stream `provenance` (missing / direct / roi / interp / genuine), `confidence` (top probability) and the five best `candidates`. It defaults to **CPU** because a second model on a GPU that is mid-training once killed a multi-hour run.
* **`/enrol` safety:** path components come from the browser, so they are filtered to a safe character set *and* the resolved path is asserted to stay inside `custom/`. Temporary upload files are always deleted. `/correct` writes the user's pick as a labelled example (`runs/corrections.jsonl`).
* **`islvit/demo.py` (205 lines)** wraps `serve.py` without editing it:
  * swaps `serve.PAGE` for the new interface (`build_page:45`, rebuilt if any UI source is newer than `ui/index.html`);
  * replaces `serve.classify` with `restrict_to_deployed:69`, which averages the views over all 262 words, zeroes the 212 non-deployed words, and takes the top word exactly as `mask50` does (renormalised for display only). Without it the live app answered from all 262 words (≈ 72.0 % instead of 85.9 %). Verified: the demo's answers agreed with the offline per-clip result on 8 of 8 held-out clips (`[report]`, Review 3 report Table 6);
  * `--sr <generator.pt>` binds the super-resolution front end into `extract`/`extract_landmarks` through `functools.partial`;
  * `--host 127.0.0.1` (default, for an SSH tunnel), `--host tailscale` (this machine's tailnet address only), or an explicit address; `--device cpu|cuda` (CPU hides CUDA from every stage including MediaPipe and the upscaler); `--all-words` (answer from all 262); a subclass of `DemoHandler` adding **read-only** `GET /samples` and `GET /samples/<file>`, which serve only files named in the manifest.

### 10.2 The interface (`islvit/ui/`)

Built into a single `ui/index.html` (≈ 4,100 lines, no external requests, works offline) by `python islvit/ui/build.py`, which concatenates `ui/src/shell.html`, `base.css`, `core.js` and the seven screens (each a `.html`, `.css`, `.js` triple). Opened without the server (`file://` or `?preview`) the page runs in a preview mode that simulates predictions and says so. State lives in `App.S` (`core.js:113`): default threshold **τ = 0.40**, **6** views.

Numbers embedded in the page are read from repository files, not typed: the 50 deployed words and **per-word recall** from `runs/f16_262w_lm_s{0,1,2}_qat4/masked50_clean.json`, the **gating curve** from `runs/gate_mixed.json`, the look-alike groups from the validation error analysis (§8 R21), and reference detection rates from the 32-frame cache (INCLUDE left 91.1 %, right 85.2 %, face 99.9 %; CISLR left 25.4 %, right 49.6 %).

| Screen | File | What it does |
|---|---|---|
| **Capture** | `screens/capture.js` | Record 3 s from the camera (browser `MediaRecorder`), drop or choose a video, or play one of ten built-in clips. The answer appears in a ring with **21 ticks (one per tracked hand joint)** lit in proportion to confidence and a notch at τ; **Accept** or **Not this word**. States: idle, reading, sure, unsure (the same rule as everywhere: confidence ≥ τ). Shows "Read in X s · 6 views · 32 frames" |
| **Not sure** | `screens/notsure.js` | Clips below τ, or rejected on Capture: the five candidates. Picking one writes a correction (`/correct`) and adds the word to the sentence |
| **Sentence** | `screens/sentence.js` | The accepted words as tiles: drag to reorder, × to remove, **Merge repeated words** (display/speak/copy/export only), **Speak** (browser text-to-speech, lights each tile), **Copy** (clipboard), **Export** (JSON of the sentence and every clip with outcome, confidence, detection share, timestamp, model, settings), **Clear** (needs a second click within 3 s). Alongside, a timeline of every clip and what became of it. No grammar or translation step: signs are isolated words |
| **Settings** | `screens/settings.js` | τ (slider and chart handle together) and number of views. The chart is the gating curve of an **older pixel-only ensemble** and is labelled as such |
| **Detection** | `screens/diagnostics.js` | Where each crop box came from (direct / ROI / interpolated / missing) per stream, per clip, over the last minute or the session. An interpolated box is a stale box, so a stream "80 % covered" can still be mostly background |
| **Vocabulary** | `screens/vocabulary.js` | Per-word recall of the shipped seed on the 472 held-out clips (about 9 to 11 clips per word, so one clip moves a word by ~10 points). Bands: reliable ≥ 90 %, patchy 50 to 89 %, weak < 50 % |
| **Enrolment** | `screens/enrolment.js` | Record three takes per word as a new signer into `custom/<signer>/<word>/takeN.webm`. The queue puts the weakest words and the look-alike groups first |

### 10.3 The demo clip set

`python -m islvit.demo_samples --count 10` (with the demo server running) builds `demo_samples/` `[local]`: ten **held-out test clips** from INCLUDE (Sridhar et al., CC-BY-4.0) transcoded to 720p H.264 with PyAV, so the file the audience sees is the file the model reads. Selection: (1) candidate = a test clip all three release models get right, one per word, spread across categories; (2) each is posted to the server and kept only if the top word is correct above τ; (3) one **hard case**, a look-alike word the model is unsure of, so the Not-sure path is shown honestly.

| Clip | Server top probability (all-262 mode, `manifest.json`) |
|---|---|
| loud / Hat / Brown / Hello / Brother / I / quiet / Dress / Grey | 91.9 / 87.0 / 94.6 / 96.8 / 96.1 / 96.9 / 95.8 / 66.0 / 87.2 % |
| **wide** (hard case) | **7.3 %**, right word still in the shortlist |

The clips illustrate the interface; they say nothing about accuracy. Evidence: `demo_samples/manifest.json` `[local]`.

### 10.4 Reaching the app from another computer

The server listens on `127.0.0.1:7860`. From the presenting computer, over the same SSH login, open a tunnel and browse to `http://localhost:7860` (this also makes the page a secure origin, which browsers require before granting camera access):

```bash
ssh -N -L 7860:127.0.0.1:7860 <user>@<this machine>
```

Alternatively, with both machines on one Tailscale network, `python -m islvit.demo --host tailscale` binds only the tailnet address.

### 10.5 What was tested, and how well it is evidenced

| Test | Method | Result | Evidence status |
|---|---|---|---|
| Screens render | Automated Chromium (Playwright), 7 screens at 1366 and 1920 px, console errors, layout measurements, transition timing | No console errors | `[report]` (Review 3 report Table 6); screenshots in `report/review3/assets/ui_*.png`; **the test harness scripts were run from a temporary working directory and are not committed** |
| Live answers | All 10 demo clips posted to the live server; 4 also played through Capture | 9 of 9 reliable clips correct at 66 to 97 %; the look-alike clip routed to Not sure with the right word first | `[report]`; `demo_samples/manifest.json` `[local]` |
| Same answers as the evaluation | Server classification code vs offline per-clip result, same clips and model | 8 of 8 agree, including the 2 the model gets wrong | `[report]` |
| Vocabulary restriction | 50-word vs 262-word answers | matches the 85.9 % setting; 262 words would give 72.0 % | `runs/f16_262w_lm_s*_qat4/masked50_clean.json` `wide_tta` vs `masked_tta` `[file]` |
| GAN front end in the app | 180p copy of a demo clip with `--sr` | read correctly | `[report]` |
| Latency | Desktop CPU, six views | 5 to 9 s per clip | `[report]`; 7.4 s on `report/review3/assets/ui_capture.png` |

---

## 11. Documents and deliverables

### 11.1 Documents in the repository

| Document | What it is | How it is produced |
|---|---|---|
| `README.md` | Public summary: headline table, leakage ladder, layout, data sources. Brought up to date on 2026-10-23 (§14.6) | hand-written |
| `docs/PRELIMINARY_MODEL_REPORT.md` (+`.pdf`) | The technical report: 19 sections, 2,411 lines; §11 holds the results chronologically, §11.13 the shipped landmark model | text hand-written; numbers from `runs/*/summary.json`; figures by `islvit/figures.py`, `figures_v2.py` |
| `docs/REVIEW_PRESENTATION_SCRIPT.md` | Speaking script for the 18-slide Review 2 deck, with Q&A, pre-review checklist, number card and glossary appendices | hand-written |
| `docs/CAPTURE_PROTOCOL.md`, `docs/capture_wordlist.txt` | Protocol for recording a signer-independent validation set: 3 signers × the 50 deployed words × 3 takes, deliberately varied room, light, clothing and distance | hand-written |
| `docs/GAN_COMPARATIVE_STUDY.{md,docx,pdf}` | GAN vs no-GAN comparison of three GAN versions (§9) | `python -m islvit.gan_report` (reads `runs/`; pandoc for DOCX; Word for PDF) |
| `docs/GAN_IMPROVEMENT_PLAN.md` | Why the first GAN stopped helping, the cheap checks, and the plan that became v3 | hand-written; numbers from `runs/gan_study/diagnostics/` |
| `docs/figures/` | 26 figures: fig01 to 09 (`figures.py`), fig10 to 18 and 20 (`figures_v2.py`), fig19 (`walkthrough.py`), fig21 to 23 (`gan_report.py`), architecture figures `figA`, `figB`, and `social_preview.png` (`make_social_card.py`) | scripts named |
| `report/case_study/` | Review 2 paper-style case study (MD, HTML, DOCX, PDF), its diagrams (`diagrams/*.drawio` + PNG) and builders (`build_report.py`, `make_assets.py`, `make_diagrams.py`) | scripts named |
| `ISL_ViT_Tiny_Presentation.html`, `ISL_ViT_Tiny_Review_Document.html`, `ISL_ViT_Tiny_Review_Pack.docx` | Earlier-review deliverables | |

### 11.2 Review 3 materials (`report/review3/`)

Review 3 asked for: the models integrated with a UI and results compared (working model with UI, 4 marks); a case-study report, plagiarism-checked (6); a presentation explaining algorithms, architecture diagram, comparisons and dataset details (6); a poster (2). **The poster was not made.**

| File | What it is | Rebuild |
|---|---|---|
| `ISL_Reader_Review3_Case_Study.{docx,pdf}` | The report: 15 pages, ≈ 7,400 words, Title / Authors with affiliation / Abstract / Keywords / Introduction / Related work / Methodology / Results / Conclusion / 37 references (Springer style). Report text is in `report_content.js` | `node report/review3/build_report.js` (npm `docx`); PDF exported with Word |
| `ISL_Reader_Review3.{pptx,pdf}` | The deck: 18 slides with speaker notes, built with native editable charts from `assets/chart_data.json`. Team 3 names and roll numbers on the title slide and every footer | `node report/review3/build_deck.js` (npm `pptxgenjs`; also needs the external theme helper `apply_theme.js`, located through the environment variable `PPTX_SKILL`); PDF exported with PowerPoint |
| `make_charts.py` | Charts c1 to c5 (model variants, GAN accuracy, 2×2 decomposition, v3 validation, leakage) and `assets/chart_data.json`; every number read from `runs/` | `python report/review3/make_charts.py` |
| `diagrams.html` | Source of diagrams d1 (application architecture) and d2 (the GAN) | rendered to `assets/d1.png`, `d2.png` |
| `assets/` | Charts, diagrams, the live-interface screenshots `ui_*.png`, figures from the case study | |

Build dependencies (`docx`, `pptxgenjs`, the theme helper) are **not** pinned in the repository (no `package.json`); the built `.docx`/`.pptx`/`.pdf` are committed. PDF export uses Word and PowerPoint through COM, so it works on Windows only.

---

## 12. Command cookbook

All from the repository root with the `islvit` environment active; `export PYTHONPATH=.`.

**Use the released model**

```bash
# one clip, 6-view TTA, CPU or GPU
python -m islvit.predict --run release/isl_vit_tiny_lm_int4_s0.pt --video path/to/clip.mp4
# add --sr runs/sr_gan/generator.pt to upscale sub-720p video with a trained generator
python -m islvit.serve --run release/isl_vit_tiny_lm_int4_s0.pt          # original built-in UI
python -m islvit.demo                                                     # the seven-screen interface (+ --host tailscale, --sr, --all-words)
python -m islvit.walkthrough --run runs/f16_262w_s0                       # trace one clip through every stage
```

**Reproduce the headline number** (needs the clean test caches, §5.4)

```bash
ISLVIT_CACHE=cache128_f32_test ISLVIT_CACHE_LM=cache_lm_f32_test \
  python -m islvit.mask50 --run release/isl_vit_tiny_lm_int4_s0.pt --out /tmp/s0.json
# expect masked_tta.top1 0.8453 (seeds 1 and 2: 0.875, 0.858; mean 0.859)
```

**Rebuild data**

```bash
python -m islvit.data.splits                                   # split CSVs with leakage assertions
python -m islvit.data.fold_val --split splits/full263__session-disjoint.csv
python -m islvit.data.crops --cache-dir cache128_f32 --crop-size 128 --frames 32 --workers 5
python -m islvit.data.landmarks --cache cache128_f32 --out cache_lm_f32 --frames 32 --workers 5
bash run_clean_test.sh                                         # the 472-clip fresh-detector test caches
```

**Train the shipped model** (the exact recipe is in `run_landmarks.sh`, `run_lm_s2_resume.sh`, `run_lm_qat.sh`)

```bash
ISLVIT_CACHE=cache128_f32 python -m islvit.train --config configs/full263_v2.yaml \
  --split-file splits/full263__session-disjoint-foldval.csv --init-from runs/isign_mim/pretrained.pt \
  --resolution-jitter 0.5 --select last --epochs 1000 --n-frames 16 --landmarks --seed 0 --tag f16_262w_lm_s0
ISLVIT_CACHE=cache128_f32 python -m islvit.qat --run runs/f16_262w_lm_s0 --epochs 300 --group 128 --seed 0 --tag f16_262w_lm_s0_qat4
python -m islvit.release --run runs/f16_262w_lm_s0_qat4 --out release/isl_vit_tiny_lm_int4_s0.pt
python -m islvit.pretrain --cache cache_isign --epochs 100 --tag isign_mim     # SSL init (8 frames by default; the iSign crop cache must be rebuilt first, it is no longer on disk)
```

**GAN study**

```bash
bash run_gan_study.sh                     # bicubic, SR-L1, v1: train, extract, score, report
GAN_ITERS=80000 bash run_gan_study_v2.sh  # v2
bash run_gan_v3.sh && bash run_gan_v3_test.sh   # v3 (hand proxy first: python -m islvit.sr.handproxy build/train)
python -m islvit.gan_report               # regenerate docs/GAN_COMPARATIVE_STUDY.* and fig21 to 23
```

**Documents**

```bash
python islvit/ui/build.py                                       # rebuild islvit/ui/index.html
python report/review3/make_charts.py
node report/review3/build_report.js && node report/review3/build_deck.js
python -m islvit.figures && python -m islvit.figures_v2         # technical-report figures
```

---

## 13. Master results index

Each row: result, value, algorithm, and the file to read it from. Tags as in §0.1.

| Result | Value | Algorithm | Evidence |
|---|---|---|---|
| Shipped top-1 / top-5 / balanced | 85.9 / 96.4 / 85.8 % | INT4 QAT landmark ViT, masked-50, 6-view TTA | `runs/f16_262w_lm_s*_qat4/masked50_clean.json` `[file]` |
| Per-seed shipped top-1 | 84.5 / 87.5 / 85.8 % | same | same |
| FP32 landmark / pixel-only | 86.5 % (87.3 / 86.9 / 85.4) / 73.7 % (75.0 / 73.7 / 72.2) | same without quantisation / without landmarks | `runs/f16_262w_lm_s*/masked50_clean.json`, `runs/f16_262w_s*/masked50_clean.json` `[file]` |
| Post-training INT4 | 83.4 % (83.1 / 85.2 / 82.0) | group-128 rounding, no training | `runs/f16_262w_lm_s*/ptq4_clean.json` `[file]` |
| All-262-word answers | 72.0 % top-1, 92.7 % top-5 | no 50-word mask | `...qat4/masked50_clean.json` key `wide_tta` `[file]` |
| Landmark gain, paired | +76/−18, +83/−21, +88/−26; p ≤ 4.6e-9 | exact McNemar on identical clips | computed from `hits_masked_tta` in the two file sets above `[file]` |
| QAT vs post-training INT4, pooled | +77/−41, p = 0.0012 | exact McNemar, pooled over seeds | `ptq4_clean.json` vs `masked50_clean.json` `[file]` |
| Parameters | 3,894,214 (landmark projections 91,008) | `ISLViT(262, 16 frames, landmarks=True)` | `islvit/models/isl_vit.py` `[file]` |
| Shipped file | 2,069,700 bytes each | `export.save_int4` container | `release/*.pt` `[file]` |
| Five-seed noise floor | 35.0 ± 1.9 % top-1 (TTA 37.5 ± 2.2) | five seeds, INCLUDE-263 session-disjoint | `runs/sd263_ssl_foldval{,_s1,_s2,_s3,_s4}/summary.json` `[file]` |
| Leakage ladder (8-frame model) | 94.5 → 29.3 → 22.1 % (INCLUDE-263) | one recipe, four split files | `runs/full263_v2__*`, `runs/sd_full263` `[file]` |
| Frame sweep, 32-frame cache | 8: 60.0, 16: 69.9 / 70.3, 24: 69.7, 32: 72.5 | frames drawn from `cache128_f32` | `runs/f{8,16,24,32}_clean_s*/summary.json` `[file]` |
| Epochs 250 → 2,000 | +7.9 | fixed budget, final weights | `runs/vocab50_s*`, `long750_s0`, `long2000_s1`, `grok_wd005` `[file]` |
| Resolution-matched CISLR pretraining | 30.4 → 39.9 % | `resolution_jitter` at fine-tune time | `runs/sd263_rj_s*`, `runs/sd263_ssl_rj_s*` `[file]` |
| iSign scale-up | 42.1 % (TTA ≈ 44.0) | same recipe, 17,896 clips | `runs/sd263_isign_rj_s*` `[file]` |
| Cross-corpus (CISLR) | 0.3 to 1.8 % | INCLUDE-trained models on 609 CISLR clips | `crosscorpus_eval.log` `[log]` |
| Encoder-freezing sweep | T-B ≈ 1 % at every scale | `backbone_lr_scale` 0 / 0.01 / 0.1 | `freeze.log` `[log]`; `runs/frz00_isign`, `frz001_isign` |
| Validation screening of landmark levers | none survives | `--lm-aug/interp/velocity/pair/wrist-vel` on 154 clips | `runs/dev_lm_*/val50.json` `[file]` |
| Gating curve (older ensemble) | 88.7 % at 75.2 % coverage (τ = 0.4) | `islvit.gate` | `runs/gate_mixed.json` `[file]` |
| GAN: learned upscaler vs bicubic at 180p | 61.0 → 75.4 to 76.0 % | ×4 generator before detection | `runs/gan_study/frozen_int4/L3_*`, `runs/gan_study_v1/frozen_int4/` `[file]` |
| GAN vs its no-GAN control | 2 of 9 pass, both at 240p | pre-registered McNemar rule | `islvit/gan_report.py` `compare` over `runs/gan_study*/` `[file]` |
| Fine-tuned on 360p, 180p | bicubic 64.4 vs v1 77.8, v2 76.9 | 150-epoch fine-tune through each front end | `runs/gan_study/retrained/`, `runs/gan_study_v1/retrained/` `[file]` |
| 180p hand detection | left found 51.0 → 68.6 (v2), 77.6 % (v3) | MediaPipe on super-resolved frames vs clean | `runs/gan_study/detection/L3_*.json` `[file]` |
| Landmark share of 180p loss | ≈ 7.7 of 10.4 points | 2×2 clean/degraded mix | `runs/gan_study/diagnostics/res_clean_pix_*`, `mirror/*` `[file]` |
| Detector levers | ≤ +0.8 points | thresholds, ROI size, multi-scale, gap fill | `runs/gan_study/diagnostics/detector_levers.json` `[file]` |
| Hand proxy | mean error 0.0215, PCK@0.05 91.4 % | ResNet-18 + soft-argmax heatmaps | `runs/hand_proxy/summary.json` `[file]` |
| Image quality at 180p | PSNR 30.07 (bicubic) to 31.83 (SR-L1) | held-out 128 px patches | `runs/gan_study*/sr_quality*.json` `[file]` |
| Interface tests | no console errors; 8/8 agreement | Playwright; server vs offline | `[report]` (§10.5) |

---

## 14. Caveats, corrections and open gaps

### 14.1 The leakage comparison: found mismatched, now corrected

**What was wrong.** The headline "98.5 % on a random split vs 75.6 % on unseen sessions (22.9 points), same model" paired two different models:

| | Random-split figure | Session-disjoint figure (old pairing) |
|---|---|---|
| Run | `runs/baseline_random_s0` | `runs/f16_262w_s0` |
| Vocabulary trained | 50 words (`vocab50__random-video`) | 262 words, masked to 50 at test |
| Epochs | 2,000 | 1,000 |
| Test clips | 200 | 472 |
| Top-1 | 98.5 % | 75.6 % |

**The matched pair** (same 50-word recipe, 2,000 epochs, 16 frames, pixel-only, iSign init) is `runs/baseline_random_s0` against `runs/f16_clean_s0` / `runs/f16_clean_s1`: **98.5 %** (one seed, 200 clips) against **69.9 / 70.3 %** (mean 70.1 %, two seeds, 472 clips), a gap of **about 28 points**. The 262-word model masked to 50 words (75.6 %) is a better model than the 50-word recipe, which is why the old pairing understated the gap.

**What was corrected** (2026-10-23): `islvit/figures_v2.py` `fig_leakage_now` (now reads both bars and the start of the honest-number trajectory from the run files) and `docs/figures/fig10_leakage_now.png`; `report/case_study/make_diagrams.py` and `diagrams/fig3_evaluation_split.{png,drawio}`; the Review 2 case study text, HTML, DOCX and PDF; the two Review 2 HTML deliverables (text, embedded figure and its alt text); the Review Pack DOCX sentence; the Review 3 report, deck, chart `c5` and `chart_data.json`. The message (the standard split greatly overstates real performance) was never in doubt; its size is about 28 points, not 22.9.

**What remains in older documents.** The Review 2 protocol ladder in the two HTML deliverables lists a take-group row (44.1 %) from a 262-word model scored on all 262 words, so that row is not a matched comparison either; and `docs/REVIEW_PRESENTATION_SCRIPT.md` and the Review Pack describe the earlier 8-frame ladder (95.7 → 46.5 → 31.3 %), which *is* matched across protocols. They are left as the record of those reviews.

### 14.2 Corrections the project made to itself

Recorded in detail in §8; in one place:

| Earlier claim | Corrected to | Where |
|---|---|---|
| INCLUDE-50 session-disjoint 35.8 % | Withdrawn: the split was built from the 943-clip subset, giving 177 take-group blocks, and train and test shared 10 of 13 real sessions. Rebuilt with corpus-wide blocks (§3.5) | technical report §6.4 |
| CISLR pretraining worth +7.7 (p < 0.0001) | +1.9 at p = 0.36 once the recipe was fixed (R6); then +9.5 once the resolution mismatch was fixed (R7) | R6, R7 |
| Pretraining jitter helps further | Rejected: four seeds, mean 37.2, sd 4.2 | R7 |
| "Data-limited, not capacity-limited" | Half wrong: the temporal axis was capacity-limited and 250 epochs hid it (+6.4) | R12 |
| Training longer is a general lever | Saturated at 16 frames (+0.6) | R11, R15 |
| Pixel-only INT4 75.6 % | Optimistic: best epoch was chosen on test; landmark QAT re-run with fixed epochs and final weights | §7.4 |
| Frame ablation U-shape on INCLUDE-50 | Instrument noise, not a result | §4.4, R3 |
| Two recorded predictions (32 frames underperform; native-16 pretraining pays) | Both wrong | R14, R15 |
| Fine-tune on the deployed objective (+2.8 on seed 0) | Unstable: seeds moved in opposite directions | R17 |

### 14.3 Results stated in documents with no committed file or script behind them (`[report]`)

* CISLR crop sharpness (variance of Laplacian 58.5 vs 27.8; box scale 173 vs 78 px).
* Retrieval 10× chance and 1-NN 1.6 % on CISLR features; the off-domain confidence statistics (mean 0.077; none above 0.4).
* The 4- and 5-member ensemble rows (75.4, 75.8 / 92.8 %; 90.7 % at 75 % coverage). The 2-seed (74.4 %) and 3-model (75.0 %) rows are in `runs/gate_f16.json` and `runs/gate_mixed.json`.
* The 262-word-masked vs 50-word-specialist pair (75.6 vs 73.3 % on the same 472 clips).
* TTA gain across eight checkpoints (+1.6 to +3.7); the individual checkpoint summaries hold `test` and `test_tta`, but no single file aggregates them.
* GMACs and CPU latency, INT8 3.74 MB, the INT4 re-quantisation weight delta (5.96e-8), the landmark/crop-box alignment check (median 0.0004 of the frame), the clean-clip byte-identity check with and without `--sr`, and landmark jitter statistics.
* All UI test outcomes (§10.5): the harness scripts are not committed.

### 14.4 Other inconsistencies and gaps worth knowing

* **Two measurements per model.** `runs/f16_262w_lm_s*_qat4/summary.json` embeds a `masked50` block from the older, detector-history-dependent cache (seed 0: 85.2 %). The headline uses `masked50_clean.json` (seed 0: 84.5 %), the cache that matches deployment (§5.5).
* **iSign counts.** 18,000 clips were extracted and 17,896 were used (`runs/isign_mim/summary.json`, `isign_pipeline.log`). Earlier documents said "~11,000 distinct source videos" (README, technical report) and 5,999 (the extractor's docstring); the log shows 18,139 distinct source videos available and 18,000 selected, one per video. All three were corrected on 2026-10-23.
* **Units.** "MB" in older sections often means MiB (§0.3).
* **Fine-tuned run summaries.** `runs/ft360_*/summary.json` show `test.n = 0` (their test clips are not in the 360p training cache); the real scores are in `runs/gan_study/retrained/` (§9.9).
* **The Settings gating curve** is from an older pixel-only ensemble, not the shipped landmark model.
* **Pre-registration.** Only GAN v1 was pre-registered; v2 and v3 were designed after earlier results were seen (judged by the same rule).
* **Not tested:** v3 fine-tuned classifier; real phone or glasses video; a second corpus for accuracy (the classifier is at chance on CISLR); latency on wearable hardware; streaming (the `<no-sign>` class and sliding window are specified but not built); signer-disjoint evaluation (no new signers recorded).
* **Binding limitation.** Every accuracy here means "on INCLUDE-like studio video"; nothing in this repository supports a claim of general ISL recognition.

### 14.5 Housekeeping

* `README.md` was updated on 2026-10-23: run and split counts (129 entries, 22 files), the layout block (`sr/`, `ui/`, `demo.py`, `report/`), a short section on the super-resolution front end, the headline reproduction command with the cache environment variables, and a link to this document.
* `test.py` in the root is an unrelated Playwright snippet and not part of the project.
* The six `run_gan_*.sh` scripts and the demo have console logs that are not tracked (§2.1).
* Source video, caches and weights are never redistributed; figures showing INCLUDE frames carry the CC-BY-4.0 attribution (Sridhar et al., ACM MM 2020, Zenodo 4010759).

---

## 15. Glossary

| Term | Meaning |
|---|---|
| **ISL** | Indian Sign Language |
| **INCLUDE** | The labelled word-level ISL corpus (4,257 clips, 262 words) |
| **MVI id** | The camera's running clip counter in each INCLUDE filename; the only handle on recording sessions |
| **take-group** | Clips of one word whose MVI ids are within 5: back-to-back near-duplicate takes |
| **session block** | One of 13 contiguous MVI ranges recovered from the whole corpus; behaves like a recording session |
| **session-disjoint** | Split protocol that holds out whole session blocks |
| **foldval** | A split with the validation clips folded into train |
| **deployed words / masked-50** | The 50 words the product answers from; scoring that zeroes the other 212 output columns |
| **TTA** | Test-time augmentation: 6 views (3 temporal phases × flip with hand streams swapped), probabilities averaged |
| **factorised ViT** | Attention within each crop, then across crops, instead of jointly |
| **stream** | One of the three crops per frame: left hand, right hand, face |
| **geometry** | A crop box's normalised centre and size, fed back because cropping deletes location |
| **provenance / source** | How a crop box was obtained: direct, ROI rescue, interpolated, or missing |
| **landmarks** | MediaPipe's 21 joints per hand and 7 upper-body points per frame |
| **SimMIM** | Masked image modelling with a mask token inside the encoder; used for self-supervised pretraining |
| **resolution jitter** | Degrading a training clip to a random lower resolution and back, one factor per clip |
| **EMA** | Exponential moving average of the weights |
| **PTQ / QAT** | Post-training quantisation / quantisation-aware training |
| **INT4, group 128** | 4-bit weights with one FP16 scale per 128 consecutive weights |
| **STE** | Straight-through estimator: gradients computed at the quantised point, applied to the full-precision weights |
| **front end** | What happens to a frame before hand detection: here, bicubic upscaling or the super-resolution generator |
| **SR-L1** | The super-resolution generator trained with pixel L1 only (the no-GAN control) |
| **SR-GAN v1 / v2 / v3** | The three adversarially trained generators (§9.5) |
| **RaGAN** | Relativistic average GAN loss |
| **hand proxy** | A differentiable network trained to read MediaPipe's hand landmarks, used as a loss for v3 |
| **PSNR / SSIM / sharpness** | Pixel-fidelity, structural-similarity and variance-of-Laplacian image measures |
| **McNemar test** | Exact paired test on the clips two models classify differently |
| **pre-registered rule** | "Helps" needs a higher mean, all three seeds agreeing, and pooled p < 0.05, fixed before the runs |
| **T-A / T-B** | INCLUDE session-disjoint test / CISLR cross-corpus test |
| **gating / coverage** | Answering only above a confidence threshold; coverage is the share of clips answered |
| **τ (tau)** | The interface's confidence threshold, default 0.40 |

### 14.6 Corrections applied on 2026-10-23

| Area | Correction | Files |
|---|---|---|
| Leakage pair | Matched recipe: 98.5 % vs 70.1 %, about 28 points (§14.1) | `islvit/figures_v2.py`, `docs/figures/fig10_leakage_now.png`, `report/case_study/*` (text, diagram, HTML, DOCX, PDF), `ISL_ViT_Tiny_Presentation.html`, `ISL_ViT_Tiny_Review_Document.html`, `ISL_ViT_Tiny_Review_Pack.docx`, `report/review3/*` (report, deck, chart, assets) |
| Stale results in the Review 2 case study | Two-seed means replaced by the three-seed results: landmark FP32 86.5 %, post-training INT4 83.4 %, shipped 85.9 % top-1 / 96.4 % top-5 / 85.8 % balanced; paired gains 76 / 83 / 88 against 18 / 21 / 26, p ≤ 5e-9 on the clean test cache; QAT cost 3.1 points | `report/case_study/ISL_ViT_Tiny_Case_Study.*` |
| Review 3 report statistical claim | "McNemar p below 1e-9 for each seed" was false on the clean cache (p = 1.2e-9, 6.7e-10, 4.6e-9); now "p ≤ 5e-9 for each seed" with the paired counts | `report/review3/report_content.js` |
| Honest-number trajectory | "from 42.1 % to 85.9 % on the same unchanged test set" mixed a take-group figure on another test set; now 51.7 % to 85.9 % on the same 472 clips. "42.1 % on the full vocabulary" (an older recipe) replaced by 64.7 % for the shipped INT4 models, 262-way, all 1,010 test clips, three seeds (`runs/f16_262w_lm_s*_qat4/summary.json` `test.top1`) | the two Review 2 HTML deliverables |
| iSign counts | 18,000 extracted, 17,896 used; 18,139 source videos available (§14.4) | README, technical report, presentation script, Review 2 and 3 documents, `islvit/data/isign_extract.py` docstring |
| README | Counts, layout, reproduction command, new section, link to this document | `README.md` |
| Local paths | A tracked diagnostic script wrote to a machine-local temporary directory; it now uses `$GAN_WORK/diag/mix`. A local-state ignore rule moved from the tracked `.gitignore` to the untracked `.git/info/exclude` | `runs/gan_study/diagnostics/run_clean_pixels_diagnostic.sh`, `.gitignore` |

Not changed, because it needs a decision or is out of scope: the wording of commit messages already pushed to `origin/main`; `docs/REVIEW_PRESENTATION_SCRIPT.md` and the Review Pack apart from the one sentence; the `[report]`-only results listed in §14.3.
