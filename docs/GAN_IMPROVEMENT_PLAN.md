# Making the super-resolution GAN actually good for this problem

*Written after the comparative study (`docs/GAN_COMPARATIVE_STUDY.md`). Numbers come from `runs/gan_study/`, `runs/gan_study_v1/` and `runs/gan_study/diagnostics/` (the clean-pixels diagnostic below; its script is `runs/gan_study/diagnostics/run_clean_pixels_diagnostic.sh`, the per-clip analysis `islvit/sr/diag_perclip.py`). Statements marked **[hypothesis]** have not been tested.*

## 1. What the GAN bought

Masked-50 top-1, 6-view TTA, mean of 3 seeds, 472 held-out clips. Clean 1080p is 85.9 %.

| Model | Input | Bicubic | SR-GAN (v2) | Gain | Gap to clean | Share of gap recovered |
|---|---|---|---|---|---|---|
| Frozen INT4 | 360p, JPEG 60 | 81.2 | 83.1 | +1.9 | 4.7 | 40 % |
| Frozen INT4 | 240p, JPEG 50 | 76.0 | 82.3 | +6.4 (+5.8 v1) | 9.9 | 64 % |
| Frozen INT4 | 180p, JPEG 40 | 61.0 | 75.5 | +14.5 | 24.9 | 58 % |
| Fine-tuned on 360p | 360p | 83.5 | 84.9 | +1.4 | 2.4 | 58 % |
| Fine-tuned on 360p | 240p | 77.9 | 83.1 | +5.2 | 8.0 | 65 % |
| Fine-tuned on 360p | 180p | 64.4 | 76.9 | +12.5 | 21.5 | 58 % |

So: **yes, it raised accuracy, by up to +14.5 points, and it closes roughly 60 % of the gap to clean video.** CISLR (a different problem) did not move: left hand 25.0 → 24.3 %.

But two facts stop this from being a finished GAN:

1. **The adversarial part adds almost nothing.** SR-GAN vs SR-L1 (the same network without the adversarial loss) is +1.5 at 240p (p = 0.022) and indistinguishable at 360p and 180p. Almost all of the gain belongs to *learned super-resolution*, not to the GAN.
2. **More GAN training did not help.** v2 (140 k adversarial iterations in total, v1 had 60 k) is not better than v1: nine paired comparisons, one reliable difference, and it points the wrong way (−1.2 at 360p, frozen INT4).

## 2. Why it stops where it stops

### Established by measurement

**(a) The remaining loss is mostly on the landmark/detection side, not the pixel side.** I kept the *clean* crops and gave the model the *degraded* landmarks (diagnostic, frozen INT4, 3 seeds):

| Input | Front end | Both degraded | Clean pixels + degraded landmarks | Clean both |
|---|---|---|---|---|
| 240p | bicubic | 76.0 | 80.6 | 85.9 |
| 240p | SR-GAN | 82.3 | 84.3 | 85.9 |
| 180p | bicubic | 61.0 | 73.7 | 85.9 |
| 180p | SR-L1 | 76.0 | 77.1 | 85.9 |
| 180p | SR-GAN | 75.5 | 78.2 | 85.9 |

At 180p with the GAN, 10.4 points are still missing. Degraded landmarks alone cost 7.7 of them; degraded pixels add only about 2.7. Bicubic lost about 12 to each. **The GAN has already repaired most of the pixel damage, and only part of the landmark damage.** A better-looking image cannot fix what the landmark extractor cannot find. (The two losses are not strictly additive; the split is indicative.)

**(b) Hand detection is still the bottleneck.** Share of hand frames genuinely detected, left/right, versus clean 91 % / 85 %:

| Input | Bicubic | SR-GAN | Missed vs clean (bicubic → SR-GAN) |
|---|---|---|---|
| 360p | 86.7 / 82.5 | 89.1 / 85.0 | 7.4 → 5.0 % |
| 240p | 71.1 / 71.2 | 83.2 / 81.3 | 20.8 → 10.2 % |
| 180p | 51.0 / 53.8 | 68.6 / 71.9 | 39.3 → 22.1 % |

Per clip, accuracy tracks how much hand presence is retained; the clips that still lose accuracy are the ones where detection still fails.

**(c) Sharpness is not what the task needs.** The GAN doubles crop sharpness over SR-L1 (32 vs 16 at 180p; originals 73) yet scores the same. The adversarial loss produces plausible texture; the classifier and the landmark extractor need *structure* (finger positions, hand outline). Texture neither adds nor removes information about it.

**(d) The discriminator is winning, so the generator is capacity-limited.** At the end of v2 the discriminator still scores real patches 0.79 and fakes 0.21 (adversarial loss 2.24; it was 0.69 at the start of v1). A 0.62 M-parameter generator against a 4.4 M-parameter discriminator cannot catch up with more iterations, which is consistent with v2 ≈ v1.

**(e) The pixel errors are small and structured.** PSNR gains over bicubic are 0.9–1.0 dB at 180p; the GAN is 0.7 dB *below* SR-L1, as expected. It is not producing a better reconstruction, only a sharper-looking one.

### Plausible but untested **[hypothesis]**

- **Information limit.** At 180p a hand is about 25–35 px across. Single-image super-resolution cannot add the missing finger detail; at best it denoises. Only extra *information* (neighbouring frames) can.
- **Training/deployment mismatch.** Training degrades with blur, resize, noise and JPEG. Real phone and glasses video is H.264/H.265: blocking, temporal smoothing, motion-compensated artefacts. A generator that never saw a codec cannot undo one.
- **32 × 32 low-resolution training patches** give the generator little context for a hand, which often spans most of a 32 px patch.
- **INCLUDE-only training data**: one studio, one camera, few signers.
- **Per-frame processing** can make landmarks flicker between frames. The measured landmark wrist errors do not show it, but I did not measure frame-to-frame jitter.
- **CISLR is not a resolution problem.** Its frames are natively 300 × 300 and the SR front end leaves detection unchanged. It is more likely framing, hand visibility or camera geometry. A 30-minute look at what CISLR frames contain would settle that, and it should happen before anyone spends GPU time on it.

## 3. What a properly built GAN for this problem looks like

The current model is a generic Real-ESRGAN-style GAN: right family, wrong objective. The problem is not "make video look sharp"; it is **"recover what the hand and face detectors and the sign classifier need from low-quality video."** Each change below targets a cause above.

### 3.1 Task-aware losses (targets 2c, 2a)

The perceptual term should measure what matters for signs, not ImageNet texture:

1. **Hand-structure proxy network.** Distil MediaPipe into a small differentiable network that predicts 21 hand landmarks (and a hand-present score) from a crop, using the clean landmark caches we already hold (about 2,845 train clips × 32 frames, every direct/ROI detection). Loss: the proxy's output on `G(lr)` should match its output on `hr`. This puts "are the fingers where they should be" straight into the objective.
2. **Frozen-classifier feature loss.** Match the ISL-ViT backbone features of SR crops to those of the HR crops. The model itself defines which differences matter.
3. **Keep a small VGG term** (weight ≈ 0.1) to stop textures from collapsing.
4. **Temporal consistency loss** on neighbouring frames (warp-and-compare), so landmarks do not flicker.

### 3.2 Real new information: video super-resolution (targets 2 information limit)

Feed the generator the target frame plus 2–4 neighbours (t ± 1, t ± 2), fused early or by deformable alignment (BasicVSR/EDVR-lite). Neighbouring frames carry sub-pixel and motion-blur information a single frame lacks. This is the only lever on the list that can recover detail rather than invent it. Cost: decode 2–4 extra frames per sampled frame, about 3–5× the SR inference time.

### 3.3 Match the real degradation (targets mismatch)

- Add **H.264/H.265 compression** (ffmpeg, CRF 23–38, random GOP) to the training degradations, applied to short clips, not single images.
- Cover the deployment resolutions explicitly: 360p, 240p, 180p, 144p; JPEG/codec quality ranges that bracket the study's.
- Keep the second-order pipeline for diversity, but oversample the settings that match the study levels.

### 3.4 Capacity and a stable adversarial game (targets 2d)

- Generator 2–4 M parameters (RRDB-lite or a deeper SRVGG with residual blocks), L1 pre-trained to convergence before any GAN step.
- Relativistic (RaGAN) loss, a multi-scale discriminator, and a two-time-scale learning-rate ratio so the discriminator does not run away; track `d_real − d_fake` and stop the GAN stage if it exceeds 0.5.
- Larger LR patches (48 × 48 or 64 × 64) for context.

### 3.5 Data

- Add the other corpora's training clips (iSign, custom signers) to the HR patch bank, so the generator sees more than one studio and signer set.
- Bias patch sampling towards hands and faces, and towards **hard cases**: clips where MediaPipe fails at 180p.

## 4. How to know it worked: the experiment ladder

Same 472 test clips, same scorer, same paired McNemar rule as the study, with one strict change: **every development decision is made on held-out validation sessions, never on the 472 test clips** (the project's validation-screening rule). The test set is touched once, at the end.

| Step | Change | Cost (RTX 3060) |
|---|---|---|
| A0 | Current SR-L1 and SR-GAN (done) | — |
| A1 | + codec degradations, + bigger G, + RaGAN; L1 pre-train then 40 k GAN iterations | ≈ 5 h |
| A2 | A1 + hand-proxy and ViT feature losses | ≈ 3 h (proxy) + 6 h |
| A3 | A2 + multi-frame input and temporal loss | ≈ 8 h |
| A4 | Best of the above, trained on all corpora, 100 k iterations | ≈ 12 h |
| Final | Front-end scoring on the 472 clips (9 caches: bicubic and SR-L1 are reused), CISLR detection, a retrained-classifier arm | ≈ 10 h |

Development metric (fast, on validation sessions): hand recall at 180p/240p, wrist error against clean, and frozen-classifier top-1 on a 100-clip validation subset, about 15 minutes per variant. Pre-registered success rule for the final model: it beats the current SR-GAN at 180p and 240p, all three seeds the same direction, pooled p < 0.05, and does not hurt clean 1080p.

### What it can and cannot reach

- **360p**: already within about 1–2 points of clean video. Little headroom.
- **240p**: 82.3 against 85.9 clean. A realistic target is 84 (about +2).
- **180p**: 75.5 against 85.9 clean. A realistic target is about 80 (about +4–5). Matching clean video at 180p is not a reasonable goal: detail that was never captured cannot be restored.
- **CISLR**: SR will not fix it. It needs the check in section 2 first.

## 5. Cheap checks to run before spending GPU time on section 3

*All four have been run; the results and the revised order are in section 7.*

These bound the ceiling and could change the order of work:

1. **Finish the 2 × 2 decomposition.** I have clean pixels + degraded landmarks. The mirror (degraded pixels + clean landmarks) needs the 180p crops re-extracted (about 30 minutes) and would say exactly how much a perfect landmark extractor would give back.
2. **Detector-side levers on the existing SR output** (no training): lower the hand confidence threshold at low resolution, enlarge the ROI rescue upscale, retry missed frames at a second scale, and smooth or interpolate landmarks over time. If these recover most of the 7.7 landmark points, the biggest win is outside the GAN, and the GAN's job becomes smaller and clearer.
3. **Look at CISLR frames** to see why its left hand is found 25 % of the time.
4. **Measure frame-to-frame landmark jitter** of SR output against clean, to confirm or rule out the flicker hypothesis.

## 6. Recommended order

1. Section 5 checks (about half a day, mostly CPU).
2. A1 (codec degradations, bigger generator, stable adversarial loss).
3. A2 (task losses) — the change most likely to separate the GAN from SR-L1.
4. A3 (video input) — the only step that adds information.
5. Final scoring and the report, with the v1/v2/A-series results all kept.

## 7. Results of the cheap checks (2026-10-11)

Diagnostics on the 472 test clips at 180p / JPEG 40 with the v2 SR-GAN front end, frozen INT4 models, 3 seeds. They bound the ceiling and choose the order of work; they are **not** model selection, and any lever adopted from them is re-validated on validation sessions first. Scripts: `islvit/sr/diag_detector.py`, `islvit/sr/diag_cislr.py`, `islvit/sr/diag_perclip.py`, and `runs/gan_study/diagnostics/*.sh`; results in `runs/gan_study/diagnostics/`.

### 7.1 The full 2 × 2: where the lost accuracy sits

Top-1 at 180p (clean on both sides is 85.9):

| | Pixels clean | Pixels degraded |
|---|---|---|
| **Bicubic, landmarks clean** | 85.9 | 79.0 |
| **Bicubic, landmarks degraded** | 73.7 | 61.0 |
| **SR-GAN, landmarks clean** | 85.9 | 83.1 |
| **SR-GAN, landmarks degraded** | 78.2 | 75.5 |

- **With bicubic** the two sides are of similar size (pixels −6.9, landmarks −12.2) and compound (together −24.9, more than their sum of −19.1).
- **With the GAN** the pixel side is almost repaired (−2.8) and the landmark side is what remains (−7.7); together −10.4, which is about their sum.
- So **about 74 % of the remaining loss is the landmark side**. A perfect landmark extractor would lift the GAN's 75.5 to 83.1. Perfect pixels alone would lift it only to 78.2 (+2.7). Improving image quality further, by itself, has a ceiling of roughly +3 points at 180p and less at 240p.

### 7.2 Detector-side levers do not fix it

Landmarks re-extracted from the same degraded, super-resolved frames with different detector settings, each scored beside clean crops (so the number isolates the landmark side; reference: 78.2 for the study's own settings):

| Variant | Top-1 | Hand coverage (clean 89.7 %) | Extra hands | Misplaced (> 5 % of frame height) |
|---|---|---|---|---|
| Study settings | 78.2 | 70.3 % | 2.6 % | 4.7 % |
| Lower confidence thresholds | 79.0 | 82.1 % | 5.1 % | 7.1 % |
| Bigger ROI upscale (256 → 512 px) | 78.4 | 68.0 % | 2.5 % | 4.4 % |
| Multi-scale ROI rescue | 78.2 | 84.5 % | 5.3 % | 10.7 % |
| All three combined | 78.7 | 92.3 % | 7.7 % | 12.0 % |
| Study settings + gap filling (≤ 4 frames) | 78.2 | 87.8 % | 8.2 % | 8.4 % |
| All three + gap filling | 78.1 | 99.2 % | 10.2 % | 14.1 % |

Coverage can be pushed past the clean figure (it then includes hands that are not there), while accuracy moves by at most +0.8. Every extra detection brings more misplaced and spurious ones. **The limit is landmark *quality*, not whether a hand was found.** (The shipped models were not trained with gap filling, so its null result here says nothing about training with it.)

### 7.3 Jitter

At 180p the wrist error behaves like occasional gross mislocalisation, not steady offset: the error trajectory's high-frequency energy is 0.079 for the SR-GAN and 0.107 for bicubic, against 0.055 for the clean wrist's own motion; the lag-1 autocorrelation of the error is about 0.49, so roughly half of it persists across frames and half does not. A temporal approach has something to work on, but gap filling alone did not help, so it needs training, not just post-processing.

### 7.4 CISLR is not a resolution problem

CISLR signers are tightly framed with large hands (median hand box 14 % of the frame width, against 8.5 % on INCLUDE), yet the pose stage reports a visible, in-frame wrist in only 22 % (left) and 42 % (right) of frames; Holistic's pose model expects a full body. When the wrist is found, the hand is found 79–85 % of the time. A hand model run directly on the whole frame (40 clips, 318 frames) finds a hand in 47 % of frames at confidence 0.3 and 59 % at 0.1, against 40 % for the pose-led path; those figures include false positives. A better CISLR front end needs direct hand detection, not super-resolution. It is outside this GAN's scope.

### 7.5 What this changes

1. **The GAN's objective must target landmark fidelity.** Nothing else on the pixel side is worth much more than +3 points at 180p; the landmark side is worth up to +7.6. The hand-structure proxy network and the frozen-classifier feature loss (section 3.1) move from "one of several ideas" to the main idea. The first evidence that it can work: the current GAN raised landmark-only accuracy from 73.7 to 78.2 *without* being trained for it.
2. **Detector tweaks are not a shortcut**, so the GAN does not get to hand its job to threshold tuning.
3. **Multi-frame input (section 3.2) moves down.** The jitter evidence is real but mixed; it is worth building only after the task losses show how far a single-frame generator gets.
4. **A complementary, non-GAN lever:** train the classifier on landmarks degraded the way the detector degrades them (misplaced and missing hands at the measured rates). The fine-tuned arm, which saw degraded video but not this explicitly, gained +1.4 over frozen at 180p.
5. **Realistic targets are lower than section 4 suggested.** With the landmark side recovered by half, 180p would land at about 79–80 (from 75.5); 240p at about 84. The 83.1 ceiling (perfect landmarks) is the hard upper bound for 180p without changing the classifier.

### 7.6 Revised order of work

1. **Hand-structure proxy network** (distil MediaPipe landmarks from the clean caches), with a held-out check that it reproduces MediaPipe on clean crops. Everything else depends on it.
2. **Generator v3:** larger generator, codec-matched degradations, stable adversarial loss, and the proxy and classifier-feature losses. Development metric on validation sessions: landmark error against clean, hand recall, frozen-classifier top-1.
3. **Ablation inside v3** (task losses on/off) so the report can say whether the task losses, and not just more capacity, made the difference.
4. **Multi-frame input** only if step 2 plateaus well below 83.
5. **Final scoring** on the 472 clips with the usual paired tests, plus the retrained-classifier arm.
