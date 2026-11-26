# Reproducing ISL-ViT-Tiny

This file takes you from a fresh clone to every number in the paper, in order. The source
videos are not redistributed (see "Data" below), so a full rebuild starts with downloading
three public corpora. If you only want to check the reported numbers, steps 1, 2 and 5 are
enough: they score the released models and the released checkpoints.

Every result in the paper is read from a file under `runs/` or `report/final/data/`. The
master index of result → file is `docs/PROJECT_CONTEXT.md` §13. The per-run recipe that
produced each checkpoint is stored in that run's `summary.json` (key `recipe`).

## 1. Environment

The results were produced on Windows 11 with an RTX 3060 (12 GB), 32 GB RAM, Python 3.11.15,
PyTorch 2.6.0 + CUDA 12.4, run from Git Bash. Linux works the same way; the scripts are plain
bash.

```bash
conda env create -f environment.yml      # pinned versions; requirements-lock.txt is the full freeze
conda activate islvit
export PYTHONPATH=.                      # every command below runs from the repository root
```

The `run_*.sh` scripts call `${PY:-python}`, so they use the active environment unless you set
`PY`. Run one training script at a time: on Windows two concurrent runs exhaust the commit limit
(WinError 1455). Every training stage is resume-safe; rerun the same command after an interruption.

## 2. MediaPipe models

```bash
mkdir -p models
curl -sL -o models/holistic_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/holistic_landmarker/holistic_landmarker/float16/1/holistic_landmarker.task
curl -sL -o models/hand_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
```

## 3. Data

Each corpus comes from its original source under its own licence. CISLR and iSign are gated on
Hugging Face: accept their terms on the dataset page and run `hf auth login` first.

| Corpus | Use | Licence | Source |
|---|---|---|---|
| INCLUDE | training and testing | CC-BY-4.0 | videos: Zenodo record 4010759; metadata: Hugging Face `ai4bharat/INCLUDE` |
| CISLR v1.5-a | cross-corpus test, CISLR pretraining | AFL-3.0 | Hugging Face `Exploration-Lab/CISLR` |
| iSign v1.1 | unlabelled pretraining (the released initialisation) | CC-BY-NC-SA-4.0 | Hugging Face `Exploration-Lab/iSign` |

```bash
# INCLUDE: metadata parquets into INCLUDE_raw/data/, then all 44 Zenodo archives (56.8 GB) at 1080p.
hf download ai4bharat/INCLUDE --repo-type dataset --local-dir INCLUDE_raw
python -m islvit.data.refetch_raw --all
# Do not use download_include_compressed.py: it transcodes to 480p and deletes the originals,
# which makes encoding quality predict 50 of the labels (docs/PROJECT_CONTEXT.md §3.3).

# CISLR: dataset.csv plus the video zip, unpacked to CISLR/CISLR_v1.5-a_videos/.
hf download Exploration-Lab/CISLR dataset.csv CISLR_v1.5-a_videos/CISLR_v1.5-a_videos.zip \
  --repo-type dataset --local-dir CISLR
unzip -q CISLR/CISLR_v1.5-a_videos/CISLR_v1.5-a_videos.zip -d CISLR

# iSign: the two video zip fragments (54 GB) and the metadata CSV.
hf download Exploration-Lab/iSign iSign-videos_v1.1_part_aa iSign-videos_v1.1_part_ab \
  --repo-type dataset --local-dir iSign_raw
hf download Exploration-Lab/iSign iSign_v1.1.csv --repo-type dataset --local-dir iSign_meta
echo "ALL PARTS DOWNLOADED" >> isign_download.log   # run_isign_pipeline.sh waits for this line
```

The split files in `splits/` are committed. `python -m islvit.data.splits` rebuilds them (seed
1337) and asserts that no clip, take group or recording session crosses between train and test.

## 4. Caches

The model reads memory-mapped crop and landmark caches, not video.

```bash
python -m islvit.data.crops --cache-dir cache128_f32 --crop-size 128 --frames 32 --workers 5   # 19 GB, about 98 min
python -m islvit.data.landmarks --cache cache128_f32 --out cache_lm_f32 --frames 32 --workers 5
bash run_clean_test.sh       # the 472 held-out test clips, re-extracted with a fresh detector per video
python -m islvit.data.crops --corpus cislr --cache-dir cache_cislr_f32 --crop-size 128 --frames 32 \
  --restrict-to splits/crosscorpus__cislr-test.csv                                           # the 609 CISLR test clips
```

## 5. Checkpoints (optional shortcut)

`runs/` holds every run's `summary.json`, `history.csv` and scoring JSON, but not its weights.
The weights (131 checkpoints, 1.9 GB) are attached to the GitHub release `v1.0`:

| Asset | Contents |
|---|---|
| `checkpoints_core.zip` | the pretraining initialisations (`isign_mim`, `isign_mim16`, `cislr_mim`, `cislr_mim_rj`), the released model's FP32 and QAT runs, the official-split runs, the GAN study's generators, hand proxy and fine-tuned classifiers |
| `checkpoints_experiments.zip` | every other run behind the ablations and the research log |

Unzip either one in the repository root; the files land at `runs/<run>/best.pt`,
`runs/<run>/pretrained.pt` or `runs/<run>/generator.pt`. With them you can score any run without
training it, and start the released recipe from the iSign initialisation without rebuilding the
iSign caches.

## 6. Training, in the order the paper's results were produced

| Step | Command | Produces | Paper |
|---|---|---|---|
| iSign extraction, crops, pretraining | `bash run_isign_pipeline.sh` | `runs/isign_mim/pretrained.pt` (17,896 clips, 100 epochs) | initialisation |
| Released model, seeds 0 and 1 | `bash run_landmarks.sh` | `runs/f16_262w_lm_s{0,1}` | |
| Released model, seed 2 | `bash run_lm_s2_resume.sh` | `runs/f16_262w_lm_s2`, `..._s2_qat4` | |
| 4-bit QAT, seeds 0 and 1 | `bash run_lm_qat.sh` | `runs/f16_262w_lm_s{0,1}_qat4` | |
| Release files | `python -m islvit.release --run runs/f16_262w_lm_s0_qat4 --out release/isl_vit_tiny_lm_int4_s0.pt` (per seed) | `release/*.pt`, 2,069,700 B each | model size |
| Held-out sessions score | `ISLVIT_CACHE=cache128_f32_test ISLVIT_CACHE_LM=cache_lm_f32_test python -m islvit.mask50 --run release/isl_vit_tiny_lm_int4_s0.pt` | `masked_tta.top1` 0.8453 (seeds 1, 2: 0.875, 0.858) | 85.9 % |
| INCLUDE's own split, both benchmarks | `bash run_official_all.sh` (or `bash run_official.sh include50 "0 1 2"`, then `full263`) | `runs/official_{include50,full263}_lm_s*{,_qat4}/official_eval.json` | 97.9 %, 98.8 % |
| Cross-corpus (CISLR) | `bash report/final/scripts/run_crosscorpus_release.sh` | `report/final/data/crosscorpus_release_s*.json` | CISLR test |
| Super-resolution front end | `bash run_gan_study.sh`; `GAN_ITERS=80000 bash run_gan_study_v2.sh`; `bash run_gan_v3.sh && bash run_gan_v3_test.sh` | `runs/gan_study*/`, `runs/sr*/` | 61.0 → 76.0 % at 180p |
| Ablations and the research log | the `run_*.sh` script named in each run's section of `docs/PROJECT_CONTEXT.md` §8 | `runs/<tag>/` | supplementary tables |

The GAN study writes its large intermediates to `$GAN_WORK` (default `gan_work/`, about 13 GB).
Three banks the v3 generator needs are built by hand; the commands are in
`runs/gan_study/work_dir_records/README.md`, along with the small records that the original work
directory held.

## 7. Paper, figures and tables

```bash
python report/final/scripts/make_figures.py
python report/final/scripts/make_example_figures.py    # needs the INCLUDE videos and caches
python report/final/scripts/make_srep_version.py        # writes report/final/srep/{main,supp}.tex
bash report/final/srep/build.sh                         # pdflatex + bibtex for both
```

## What to expect

Training seeds the Python, NumPy and PyTorch generators, but does not force deterministic CUDA
kernels, so a retrained model matches the reported one to within seed-to-seed variation, not
bit for bit. The spread is reported with every multi-seed result: on the held-out sessions the
three released seeds score 84.5, 87.5 and 85.8 %. Scoring a released file or checkpoint is
deterministic and reproduces the stored JSON exactly.
