# ISL-ViT-Tiny — Review Presentation Speaking Script

**Deck:** `ISL_ViT_Tiny_Complete_Review_Presentation_FINAL_FIXED.pdf` (18 slides)
**Course:** 23CSE473 — Neural Networks and Deep Learning · Group 03
**Presenters:** Dhrusheek Rishi Menon · Nethi Kushala Kumar · Yalamanchi Kushal
**Target runtime:** 20 minutes of speaking + 8–10 minutes of questions

Every number in this script was read back from the repository — `runs/*/summary.json`,
`runs/gate_*.json`, `runs/*/per_class.json`, `docs/PRELIMINARY_MODEL_REPORT.md` — not
from the slides alone. Where the deck and the repo disagree, the script says so and
gives you the safe thing to say.

---

## How to use this document

Each slide has five parts:

| Part | What it is for |
|---|---|
| **Screen anchor** | The one thing to point at, so you are never reading the slide aloud |
| **Script** | Say this. Written to be spoken, not read silently |
| **Emphasise** | The single sentence that must land if nothing else does |
| **Do not say** | Claims the repo does not support — these are how you lose a viva |
| **If asked** | Anticipated questions with grounded answers |

Two rules that hold across the whole talk:

1. **Never quote an accuracy without its protocol.** "68.9 %" is meaningless in this
   project. "73.3 % top-1 on the 472-clip session-disjoint INCLUDE-50 test set" is a
   claim. The panel will be listening for exactly this.
2. **The story is not "we built a model." It is "we found out that the benchmark was
   lying, and then we improved the honest number by 24 points without touching the
   architecture."** Every slide should serve that arc.

### Speaker handoffs

| Slides | Speaker | Why |
|---|---|---|
| 1–3 | Dhrusheek | Framing, the literature, and the evaluation gap — their contribution area |
| 4–5 | Kushal (Yalamanchi) | Pipeline and modules — they built the crop front end |
| 6–9 | Kushala (Nethi) | Metrics, architecture, algorithm, hyperparameters — the model |
| 10–11 | Dhrusheek | Results and protocol — back to the evaluation story |
| 12–14 | Kushal | UI, standard paper, competitive landscape |
| 15–18 | Kushala | Novelty, contributions, and the denoising extension |

---

# SLIDE 1 — Title

### Screen anchor
The subtitle: *"A compact factorised Vision Transformer for Indian Sign Language
recognition."*

### Script

> Good morning. We are Group 3 — Dhrusheek, Kushala and Kushal — and our project is
> **ISL-ViT-Tiny**: a compact factorised Vision Transformer for Indian Sign Language
> recognition, built under Dr. Bagyammal's guidance.
>
> Before the agenda, one sentence about what this project actually turned into,
> because it is not the project we set out to do.
>
> We set out to build a small sign-language model for a wearable. We did that — it is
> **3.76 million parameters, 0.82 GMACs per clip, 3.7 megabytes packed in INT8, and it
> runs in 12 milliseconds on a desktop CPU.**
>
> But partway through we found that the standard way this dataset is split lets a model
> memorise its way to 95 percent. When we removed that, our own number fell from
> 95.7 to about 31. **The substantive work in this project is what we did about that
> gap** — and the honest number is now 75.8 percent, from a model whose architecture
> never changed.
>
> So the talk has two threads: the engineering, and the measurement. Everything is in
> the repository on the slide — all the code, all 21 split definitions, and the
> per-run results files for all 81 runs.

### Emphasise
The pivot: this is a *measurement* project as much as a modelling project. Say it in
the first thirty seconds so the panel hears every later number correctly.

### Do not say
"We achieved 95.7 % accuracy." Even as an opener, even as a hook. You will spend the
rest of the talk walking it back.

### If asked
- **"Is the repo public?"** Yes — `github.com/not-good-keeper/compact-vit-sign-language`.
  Code, configs, all split definitions and `summary.json` / `history.csv` for every
  run. Source video, crop caches and checkpoints are excluded — roughly 128 GB — but
  every one of them is regenerable from a `run_*.sh` script in the repo.
- **"Who did what?"** Slide 16 breaks it down; briefly, Dhrusheek owns the evaluation
  protocol, Kushala owns the model, Kushal owns the data pipeline and deployment.

---

# SLIDE 2 — Problem Statement, Objective & Motivation

### Screen anchor
The purple motivation box at the bottom right: **51.7 % → 75.8 %, inside 3.76 M
parameters, 0.82 GMAC, 3.70 MB.**

### Script

> The task, stated precisely: **given a short RGB video clip containing exactly one
> Indian Sign Language sign, output the correct word from a fixed vocabulary — on a
> wearable device, offline.**
>
> Three constraints in that sentence do real work. *Isolated word*, not continuous
> translation — one clip, one label. *Offline* — no server round trip, which is a
> privacy requirement as much as a latency one, because a sign-language user's
> conversation is their conversation. And *wearable* — which pins the compute budget
> before we write a line of model code.
>
> Now, why this is hard, and these are measured, not assumed.
>
> **Low data.** 262 classes with a median of 15 clips each. That is about fifteen
> examples per word. Any model with real capacity will memorise that.
>
> **The hand is tiny.** In a 1920-by-1080 frame the signing hand spans roughly 190
> pixels. If we fed the full frame to a 64-pixel model input, the hand would occupy
> about **six pixels**. There is no architecture that recovers a handshape from six
> pixels — which is why the crop front end on the next slides is not an optimisation,
> it is a precondition.
>
> **Motion blur.** At 25 frames per second a signing hand smears. Full-frame MediaPipe
> finds a hand in only about 79 percent of sampled frames, which is why we built a
> two-stage detector.
>
> **Location is linguistic.** The same handshape at the forehead and at the chest can
> be two different words. Hold that thought — it comes back on slide 7, because
> cropping to the hand *deletes* that information and we have to put it back
> explicitly.
>
> **And session variation.** Room, clothing, lighting and signer all change between
> recordings. This is the one that turned into the main result.
>
> So the objective: build an isolated-word ISL recogniser that is both computationally
> small **and** evaluated under a protocol that does not permit memorisation. The four
> sub-objectives on the left are the four things we had to do to get there —
> reconstruct the session structure, design the factorised model, improve the honest
> number with controlled interventions, and ship confidence gating so an uncertain
> prediction becomes an explicit "not sure" rather than a confident wrong word.

### Emphasise
The six-pixel calculation. It is the most concrete justification for the entire
architecture and it takes eight seconds to say.

### Do not say
"Sign language recognition is a solved problem for large models." You do not need the
strawman, and a panelist who works in the area will push back.

### If asked
- **"Why isolated words rather than continuous signing?"** Continuous ISL needs a
  segmentation and alignment stack on top of recognition, and the aligned corpora do
  not exist at the scale we would need. We built the scaffold for it — `islvit/slt.py`
  is a preliminary encoder-decoder path — but nothing in this report depends on it.
- **"Where does 51.7 % come from?"** That is our own 250-epoch, 8-frame baseline on the
  strict session-disjoint 50-word test set. It is our starting point, not a published
  number — we deliberately compare against ourselves so the test set never changes.
- **"Why 262 classes and not more?"** That is the entire INCLUDE vocabulary. Slide 11
  covers the corpus.

---

# SLIDE 3 — Literature Survey (15 papers)

### Screen anchor
The red line at the bottom — the joint gap statement. Land the slide on that sentence.

### Script

> We surveyed fifteen journal papers, and we organised them by the three things our
> system has to be at once, rather than chronologically — because the gap we are
> claiming is a gap in the *intersection*, not in any single column.
>
> **Set A, edge and compression.** TinyMSLR reaches 99 percent under 2.7 million
> parameters with 24-millisecond CPU inference. Dynamic Kannada SLR gets 94 percent in
> a 1.1-megabyte model at 16 milliseconds on a phone. These prove the deployment
> target is achievable. But TinyMSLR's benchmark is twenty controlled classes with no
> signer-disjoint split, and the Kannada work takes *landmarks* as input, not RGB —
> which means it inherits the landmark extractor's failures and cannot see anything
> the extractor does not encode.
>
> **Set B, video transformers.** The PLOS ONE comparison — which is our chosen standard
> paper, slide 13 — evaluates VideoMAE, ViViT and TimeSformer on word-level sign
> datasets and reaches 96.9 percent. That paper is *why* our model is factorised. But
> every backbone in it is Kinetics-scale: hundreds of megabytes. Nothing in that
> family fits a wearable.
>
> **Set C, evaluation and signer independence.** This is the column that shaped the
> project. "Beyond Perfect Scores" in Technologies 2026 reports drops of **15 to 83
> points** when a random split is replaced with a structure-aware one. Isharah and
> TSL-ONE-S build signer-independent benchmarks. But Isharah is Saudi sign language
> and ships signer identities explicitly; TSL-ONE-S is Thai. **For ISL, no corpus
> ships signer identity at all** — which is precisely the problem we had to solve by
> reconstructing sessions from camera capture IDs.
>
> So the joint gap: **no surveyed work combines an RGB Vision Transformer, Indian Sign
> Language, wearable-scale deployment, and leakage-controlled session-disjoint
> evaluation in one system.** Each column exists. The intersection does not.

### Emphasise
The gap is in the *intersection of four properties*, and you can name all four. A
panel testing whether a survey was real will ask you to name a limitation of a specific
paper — every cell on that slide has one written under it, so read the gap line, not
the result line.

### Do not say
"Nobody has done sign language recognition with transformers." Obviously false; the
middle column of your own slide refutes it.

### If asked
- **"Which paper is closest to yours?"** The PLOS ONE video-ViT comparison on task and
  model family, TinyMSLR on deployment budget. Neither does both, and neither uses a
  leakage-controlled split.
- **"What does 'signer-disjoint' mean versus your 'session-disjoint'?"**
  Signer-disjoint means no signer appears in both train and test. We *cannot* do that
  on INCLUDE because signer IDs are not shipped. Session-disjoint is our proxy: whole
  recording sessions are held out, and consecutive takes within a session share
  signer, room, clothing and framing. It is strictly weaker than signer-disjoint and
  we say so on slide 11 and in the limitations.
- **"Did you reproduce any of these?"** No, and we do not claim to. We reproduce the
  *published INCLUDE range* — 94–95 % under a random split — which is the relevant
  comparison, and then show what it is made of.

---

# SLIDE 4 — Overall Application Architecture

### Screen anchor
Trace the four numbered boxes with your hand in order: 1 offline → 2 split → 3
training → 4 online. Do not let your eye wander to the sub-boxes while talking.

### Script

> This is the whole system on one slide. Four stages; I will take them in order and
> flag the two places where a design decision is doing real work.
>
> **Stage 1, offline dataset preparation — run once.** 4,257 INCLUDE clips at
> 1920-by-1080. We sample 32 frames per clip, normalise to a 720p working height, and
> run MediaPipe Holistic for pose, face and hands on the full frame.
>
> Here is the first decision. Full-frame detection succeeds **81.4 percent** of the
> time for the left hand. For the missed frames we do not give up and we do not fake
> it — we take a region of interest from the pose wrist and elbow, upscale it to 256
> pixels, and re-run a dedicated HandLandmarker on just that patch. That **rescues 10
> to 15 percent** of frames that full-frame detection alone would have lost. Only if
> *that* fails do we carry forward the nearest real detection — and critically, we
> **flag that crop as unreliable**, and the flag reaches the model as a token. We
> never present a guess to the model as if it were a detection.
>
> The output is three crops per timestep at 128 pixels — left hand, right hand, face —
> written to an 18.7-gigabyte memmap cache along with each crop's box geometry and its
> provenance code. That extraction ran in 98 minutes with **zero failures**.
>
> **Stage 2, protocol construction.** This is the part that is unusual. INCLUDE ships
> no signer identity, but every filename carries a camera capture ID — `MVI_2978`,
> `2979`, `2980`. Consecutive IDs mean consecutive takes, which means the same signer
> in the same room in the same clothes. We parse those IDs, group takes whose IDs are
> within five of each other into **1,334 take-groups**, then cluster globally into
> **13 session blocks**, and build four protocols on top. Every split is written with
> **automated assertions** that fail loudly if a session or a take-group straddles the
> train/test boundary.
>
> **Stage 3, training.** Self-supervised pretraining on iSign — 18,000 clips, masked
> crop reconstruction — then supervised training: 2,000 epochs, AdamW, cosine
> schedule, EMA, bfloat16. 81 runs recorded on disk, each with its full config and
> history.
>
> **Stage 4, online inference.** And this is the second decision worth flagging: the
> deployment path **calls the same preprocessing functions as the offline cache
> builder**. `islvit/serve.py` imports `extract` and `views` from `predict.py` — the
> identical code path. That is deliberate. Writing a separate inference preprocessor
> is the standard way a model that scores well offline fails on a real clip, and we
> removed the possibility structurally rather than by testing for it.
>
> Then INT8 inference at 12 milliseconds, six-view test-time augmentation, and a
> confidence gate: above threshold we emit a word, below it we emit "not sure."

### Emphasise
Two things: **the unreliable flag reaches the model as a token** (nobody expects that),
and **train/serve share one preprocessing code path** (an engineering panel will
recognise this as a mature decision).

### Do not say
Do not read the percentages off the boxes. You have five numbers to say on this slide
— 81.4, 10–15, 18.7 GB, 98 minutes, 12 ms — and the rest is on screen for them.

### If asked
- **"Why 32 frames if you only use 16?"** Slide 9 and slide 10 both cover it, but the
  short answer is **headroom**. Feeding 16 frames sampled from a 32-frame cache beats
  16 sampled from a 16-frame cache by **3.2 points**, with the model, the input and
  the inference cost all identical. The only thing that changes is how much choice the
  sampler has about *where* those 16 frames land. Depth is paid for in disk, not in
  device compute.
- **"How do you know the crops are right?"** We looked at them, and that is not a
  joke — an earlier version of this pipeline synthesised fallback boxes from pose
  geometry and reported **100 percent coverage**. Rendering the crops showed they were
  shirt fabric, the studio wall, and blurred torso. MediaPipe reports optimistic
  visibility for occluded hands. Coverage was measuring box *existence*, not box
  *correctness*. That failure is why every crop now carries a provenance code.
- **"What is the memmap for?"** So that temporal and spatial augmentation are free at
  training time — no video decoding, no MediaPipe in the training loop, just array
  indexing. It is what makes 2,000-epoch runs affordable on one RTX 3060.

---

# SLIDE 5 — Module Details

### Screen anchor
The three purple-bordered cards at the bottom. Those are the ones you will explicitly
scope out as scaffolding.

### Script

> Nineteen modules; each file has one defined responsibility, and the same pipeline
> code serves the offline and online paths. I will not read all nineteen — four
> matter, and three need a caveat.
>
> **`crops.py`** is the two-stage MediaPipe pipeline with per-crop provenance and
> geometry. **`splits.py`** does capture-ID grouping, session-disjoint construction,
> and the leakage assertions — it is the module that makes every other number in this
> talk meaningful. **`train.py`** is the supervised loop with EMA, augmentation and
> checkpointing. **`isl_vit.py`**, under `models/`, is the architecture on slide 7.
>
> Around those: `pretrain.py` for SimMIM-style masked self-supervision, `tta.py` for
> the six deterministic views, `gate.py` for the accuracy-versus-coverage analysis,
> `export.py` for INT8 quantisation and *measured* model size, and `eval.py` to score
> any checkpoint on any split.
>
> The three cards at the bottom are honest scoping. **`distill.py`** is a
> teacher-to-student harness, **`slt.py`** is a preliminary encoder-decoder path
> toward continuous ISL, and `report.py` and `figures.py` generate every table and
> figure in our technical report directly from the run files. **None of the reported
> accuracy in this presentation comes from distillation or from the translation
> scaffold.** They are built, they run, they are not load-bearing.
>
> Supporting that: 13 YAML configs, 21 split definitions, 25 experiment scripts, and 81
> recorded runs. Every accuracy figure we quote today is read out of a
> `summary.json`; none of them is transcribed by hand from a log.

### Emphasise
"Every accuracy figure is read out of a `summary.json`, not transcribed from a log."
That sentence is the difference between a project that can be audited and one that
cannot, and panels notice it.

### Do not say
Do not let `distill.py` or `slt.py` sound like results. Scope them out loud, in the
same breath you introduce them.

### If asked
- **"Show me that a number traces back."** `runs/f16_clean_s0/summary.json` has
  `test.top1 = 0.6992` and `test_tta.top1 = 0.7331`, alongside the exact split file,
  cache, frame count, seed and full recipe that produced it. That is the 73.3 % on
  slide 10.
- **"Why is `figures.py` a module and not a notebook?"** Because a notebook's output
  can drift from the data that produced it. Regenerating the report regenerates the
  figures from the run files.

---

# SLIDE 6 — Performance Metrics

### Screen anchor
The subtitle line: *on INCLUDE-50, top-1 is 17 points below balanced accuracy for the
same predictions.*

### Script

> Nine metrics, and the reason there are nine rather than one is the line under the
> title, which I want to take slowly because it is the justification for most of this
> table.
>
> On our INCLUDE-50 session-disjoint test set, for **the same predictions**, top-1
> accuracy reads 14.3 percent and balanced accuracy reads 31.3 percent. Seventeen
> points apart, same model, same clips.
>
> That is not a paradox, it is the composition of the test set — and it is
> *adversarial*, not merely uneven. When you hold out whole recording sessions from a
> 943-clip corpus, the classes that end up best represented in test are the ones worst
> represented in train. We measured that correlation: **minus 0.73.** The ten classes
> with the most test clips account for 48 percent of the test set, and they are
> exactly the ten the model had the least opportunity to learn. Top-1 is therefore
> dominated by the hardest, least-trained classes. **Balanced accuracy — the mean of
> per-class recalls — weights every word equally, and on this benchmark it is the only
> interpretable metric.**
>
> So: **top-1** for headline clip-level correctness. **Top-5** because it is what a
> wearable UI actually shows — a shortlist the user picks from, and we reach 92.8
> percent there. **Balanced accuracy** for the reason just given. **Macro F1** to
> expose per-class failure structure.
>
> Then three metrics that exist because this is a product, not a benchmark entry.
> **Selective accuracy** — how often we are right *when we choose to answer*.
> **Coverage** — what fraction we answer at all. **Yield** — the product, the fraction
> of clips that are both answered and correct, which is what the user actually
> experiences. At a threshold of 0.4 we answer 75 percent of clips at 88.7 percent
> accuracy, for a 66.7 percent yield.
>
> And the one that governs how we read all the others: the **seed noise floor.** We
> ran five seeds of one configuration. Top-1 varies by 1.9 points standard deviation
> from the seed alone, which makes the standard deviation of a *difference between two
> runs* about 2.7 points. **Any single-run comparison below roughly 5 points in this
> project is unresolved, not a result** — and we applied that retrospectively and
> withdrew two of our own earlier claims because of it.

### Emphasise
The minus-0.73 correlation. It converts "we prefer balanced accuracy" from a
preference into a measurement, and it is the single most defensible thing on the slide.

### Do not say
"We use balanced accuracy because it is fairer." True but weak. Use the correlation.

### If asked
- **"Then why report top-1 at all?"** Because it is what every paper we compare
  against reports, and because on **INCLUDE-263** the two metrics agree — 22.1 % top-1
  against 25.2 % balanced. The divergence is specific to the 50-word subset, which is
  too small to admit a balanced held-out session set. Every claim requiring us to
  resolve a small difference is made on 263.
- **"Is 2.7 points a standard deviation or a confidence interval?"** Standard
  deviation of the difference of two single runs, derived from five seeds of one
  config: sd 1.9 per run, times root-two.
- **"Is the gating threshold validated?"** No, and we say so explicitly in the report.
  τ = 0.4 was chosen on the test set, so the accuracy-at-coverage figure is optimistic
  as a forward-looking estimate. We report the *shape* of the curve as a
  characteristic of the model, not one row as a product guarantee. A deployed system
  needs that threshold set on a held-out calibration set.

---

# SLIDE 7 — Deep Learning Architecture

### Screen anchor
The measured-cost box at the bottom right. Specifically the line **"naive joint ViT
would use 408 tokens → ≈ 48× cost."**

### Script

> ISL-ViT-Tiny. 3.76 million parameters, and the design follows ViViT's
> factorised-encoder variant sized down hard. Three things to explain: the
> factorisation, the token conditioning, and why the width is 192.
>
> **The factorisation, and why it is the reason this project is possible at all.**
> A naive video transformer attends jointly over every patch of every frame. At the
> configuration in the cost box — 8 timesteps, 3 streams, 17 tokens per crop — that is
> **408 tokens in one sequence**, and attention is quadratic, so 408-squared. Instead
> we factorise. **Stage A** attends *within* a single crop: 17 tokens, weights shared
> across all 24 crops. **Stage B** attends *across* crops: 25 tokens. So instead of
> O of 408-squared we pay O of 17-squared plus O of 25-squared. That is not a tuning
> choice — it is what makes the model fit an edge budget at all.
>
> [Deployed, at 16 frames, the same arithmetic is 16 × 3 × 17 = 816 tokens naive,
> against 17 spatial and 49 temporal.]
>
> **Stage A in detail.** A Conv2d patch embedding, 3 to 192 channels, kernel and stride
> 16, which on a 64-pixel crop gives a 4-by-4 grid — 16 patches. Add a CLS token and a
> learned positional embedding: 17 tokens of width 192. Four transformer blocks, three
> heads, MLP ratio 4, pre-norm, DropPath 0.1. LayerNorm, take the CLS. One 192-vector
> per crop.
>
> **Now token conditioning — this is our own contribution and it matters most.**
> Cropping to the hand solves the six-pixel problem from slide 2, but it **deletes
> where the hand is**, and sign location is a defining linguistic parameter — the same
> handshape at the forehead and at the chest are different words. So we feed the box
> geometry back in explicitly: centre x, centre y, and size, through a small MLP.
>
> The detail that matters is that **the last layer of that projection is
> zero-initialised.** At step zero the geometry branch contributes exactly nothing, so
> the model starts as a pure appearance model built on pretrained ImageNet features,
> and learns to use location *gradually*, as a refinement. Initialising it normally
> would let three raw coordinates swamp a 192-dimensional pretrained feature early in
> training, when the gradient is largest.
>
> We also add a stream embedding — which of the three crops this is — a time embedding,
> and a **missing embedding** wherever the detector never actually found a hand. That
> is the reliability flag from slide 4 arriving at the model. A never-detected stream
> gets a black crop, and telling the model "this is absent" beats making it infer
> absence from all-zero pixels, because **the absence of a hand is itself
> linguistically informative** — many ISL signs are one-handed.
>
> **Stage B** prepends its own CLS to the 48 conditioned tokens, runs four more blocks
> attending across time *and* across streams jointly, and the CLS goes to a linear
> head.
>
> **Why width 192.** It is pinned. 192 with 3 heads is exactly DeiT-Tiny's geometry, so
> ImageNet weights load into Stage A directly with no projection layer — we only
> bicubically resize the position embeddings from their 14-by-14 grid to our 4-by-4.
> With a median of fifteen clips per class, that ImageNet initialisation is the single
> biggest lever we have against the data scarcity.
>
> Measured cost, and *measured* is the operative word: 3,760,754 parameters, 94.6
> percent of them in the eight transformer blocks. 0.824 GMACs per clip, of which
> Stage A is about 94 percent. 12 milliseconds per clip on a desktop CPU. And **3.70
> megabytes packed INT8 — weighed as a file, not computed as parameters times one
> byte.**

### Emphasise
The zero-initialised geometry projection. It is the most sophisticated single decision
in the architecture, it is genuinely ours, and it has a crisp mechanical justification.
Rehearse that paragraph until it is fluent.

### Do not say
"We invented a new transformer." You did not, your own slide 15 says so, and claiming
it invites a hostile follow-up. The novelty is the conditioning and the evaluation, and
that is a real claim.

### If asked
- **"Where does the 48× come from?"** ⚠ **Do not improvise this one.** The asymptotic
  claim is unambiguous and correct: quadratic in 408 versus quadratic in 17 and 49
  separately. The *multiplier* is not cleanly derivable from the slide. Counting the
  spatial encoder once, (17² + 25²)/408² ≈ 0.005, i.e. ~180×. Counting it the 24 times
  it actually runs, (24·17² + 25²)/408² ≈ 0.045, i.e. ~22×. Neither is 48. Say **"the
  attention cost drops by well over an order of magnitude — the exact multiplier
  depends on whether you amortise the shared spatial encoder, and I can show you the
  calculation"** and move on. See Appendix B item 7.
- **"Why is INT8 size 3.70 MB and not 3.76 MB?"** Because we measured it rather than
  assuming. `quantize_dynamic` converts `nn.Linear` only — that is 95.7 % of our
  parameters. The remaining 4.3 % — the patch-embedding Conv2d, the LayerNorms, and
  the position and time embeddings — stay FP32 at four bytes each, and quantised
  tensors also carry per-channel scales and zero-points. Those effects run in opposite
  directions. The number on the slide is the size of the file on disk.
- **"Why four blocks each and not six or two?"** Balanced parameter allocation between
  the two stages, and it keeps head dimension at 64, which is the standard efficient
  value. We tested capacity increases directly and they were flat or negative — slide
  10.
- **"Why not a CNN?"** Two reasons. The pretrained-initialisation story is stronger for
  ViTs at this width, and more importantly the factorised attention is what gives us
  an explicit, cheap place to *inject* the geometry and reliability tokens. In a CNN
  there is no natural token to add a 3-vector to.
- **"Does the geometry projection actually get used?"** It has to be learned from zero,
  so the fact that the model trains to 73 % at all with it in place means gradient is
  flowing through it. We do not have an ablation with it removed at the final recipe —
  that is a fair criticism and it is in our future work.

---

# SLIDE 8 — Algorithm Procedure (Mathematical)

### Screen anchor
Step 6 on the left (highlighted purple) and step 14 on the right (highlighted red).
Those two are yours; the other twelve are standard.

### Script

> Fourteen steps. Twelve are standard transformer training and I will move through
> them quickly; **six and fourteen are ours** and I will slow down.
>
> **Forward pass.** Step one, sample 16 frames from the 32-frame cache using uniform
> segments with a random offset inside each — that is the temporal jitter that the
> cache depth buys us. Step two, crop and augment; note that our horizontal flip
> **swaps the left and right hand streams and mirrors the box geometry**, because a
> mirrored right-dominant signer is a left-dominant signer, and flipping the pixels
> without swapping the streams would be physically incoherent. Step three, patch
> embed. Step four, spatial multi-head self-attention. Step five, take the LayerNormed
> CLS as the crop token.
>
> **Step six is the conditioning equation, and it is the architectural claim in one
> line.** The token for timestep t, stream s is the crop embedding, **plus** a stream
> embedding, **plus** a time embedding, **plus** the geometry projection of that
> crop's box, **plus** a missing embedding gated by the indicator that the box was not
> genuinely detected. Every term after the first is information that cropping
> destroyed or that the pipeline knows and the pixels do not.
>
> Steps seven and eight: temporal encoder over that conditioned sequence, linear head,
> softmax.
>
> **Training.** Cross-entropy with 0.1 label smoothing. AdamW at 5e-4, with the
> **spatial encoder's learning rate scaled to one tenth** — we refine the pretrained
> encoder, we do not erase it. Cosine schedule with warmup.
>
> Step eleven, EMA, and there is a fix embedded here. A fixed decay of 0.999 means the
> average is dominated by initialisation for roughly the first thousand steps, which in
> our short 50-word epochs is a large fraction of training. So we use a **warmup
> schedule** — decay starts near zero and ramps to 0.99 — and the EMA tracks from step
> one.
>
> Step twelve, checkpoint selection, and this is a deliberate departure. **We take the
> final EMA weights rather than the best-validation checkpoint.** Under session-disjoint
> splits, validation is carved from the training sessions, so it is not
> session-disjoint from training and is therefore not a clean model-selection signal.
> Selecting on it selects for the wrong thing. We fold validation into training and
> take the last epoch.
>
> Step thirteen, test-time augmentation: three temporal phases by two flip states, six
> views, **probabilities averaged, not logits**. That is deliberate — the views are
> alternative observations of one clip, so a view that is confidently right should
> outvote one that is barely undecided. Averaging logits lets a single large negative
> logit dominate.
>
> **Step fourteen, ensemble and gate.** Average across members, then: if the maximum
> averaged probability is at least 0.4, emit the word. Otherwise emit "not sure." The
> abstention is a first-class output, not an error state.
>
> And the box along the bottom: protocol construction is a **separate algorithm**, and
> it runs before any of this. Group the capture IDs, cluster into 13 session blocks,
> search for a test-block subset that keeps enough classes evaluable, and then assert
> that no session and no take-group straddles the boundary.

### Emphasise
Step 12. "Validation is not a clean selection signal under this protocol, so we do not
select on it" is a subtle, correct, easily-defended decision and very few
undergraduate projects get it right.

### Do not say
Do not attempt to read the LaTeX-ish notation aloud symbol by symbol. Say what each
equation *does*.

### If asked
- **"Isn't taking the last epoch risky?"** It would be if we were undertrained or
  unstable. We are neither — 2,000 epochs at 16 frames is past saturation; doubling to
  4,000 buys +0.6 points, which is inside the 2.7-point noise floor. The loss curve is
  flat at the end and the EMA smooths what is left.
- **"Why 0.4 for the threshold?"** It is the knee of the curve. At 0.3 we answer 88 %
  at 80.6 % accuracy; at 0.4 we answer 75 % at 88.7 %; at 0.5, 67 % at 92 %. 0.4 gives
  the highest yield among the high-accuracy operating points. And per slide 6, it is
  chosen on test, so it is an illustrative operating point rather than a validated one.
- **"Why six TTA views and not more?"** Nine or more was not measurably better in a
  spot check and costs proportionally more. Six is three phases times two flips, which
  spans both axes the model was trained to be invariant to.

---

# SLIDE 9 — Hyperparameter Details with Justification

### Screen anchor
The **Justification / measured evidence** column. Say out loud that every cell in it is
a measurement, not a citation.

### Script

> Twelve hyperparameters, and I want to point at the right-hand column before any of
> the values, because that column is the point of the slide: **every justification here
> is a number we measured on our own test set, not a value copied from a paper.**
>
> The ones that carry weight.
>
> **Frames: 16, sampled from a 32-frame cache.** Going 8 to 16 is worth **+6.4 points**
> — the largest single input-side move in the project. Going to 32 costs roughly double
> the inference and ties after TTA, so 16 is the deployable choice.
>
> **Cache depth 32 — and this is the subtle one.** Sixteen frames drawn from a
> 32-frame cache beats sixteen drawn from a 16-frame cache by **+3.2 points at
> identical inference cost.** Same model, same input size, same latency. The only
> difference is that the sampler has somewhere to jitter. We pay for that in disk, not
> on the device.
>
> **64-pixel input from a 128-pixel cache.** 64 is the cheapest input we tested and it
> tied with everything larger — resolution is genuinely not the bottleneck. But
> sourcing those 64-pixel crops from a 128-pixel cache rather than an 80-pixel one is
> worth **+4.4 points**, because it gives spatial augmentation room to work.
>
> **Three streams — left hand, right hand, face.** Removes background, clothing and
> furniture as class cues while keeping the non-manual facial information, which in
> ISL is grammatically meaningful.
>
> **Width 192, four plus four blocks, three heads** — slide 7; pinned for DeiT-Tiny
> loading.
>
> **2,000 epochs.** Going from 250 to 2,000 was worth **+7.9 points** at 8 frames. We
> were simply undertraining everything for the first half of this project.
>
> **Backbone learning-rate scale 0.1.** We swept it: 0.0 frozen, 0.01, and 0.1. Scores
> were 31.8, 38.7 and 41.8. The pretrained encoder should be refined, not frozen and
> not erased.
>
> **Augmentation** — jitter, flip-with-swap, ±0.4 colour, grayscale 0.15, stream
> dropout 0.15, mixup alpha 0.2. Every one of those is a *physically coherent*
> invariance for this task. And we tested the alternative: a strong-augmentation bundle
> with cutmix, random erasing and DeiT-strength mixup was a **top-1 null at both frame
> counts and left the model measurably worse calibrated.** Not adopted.
>
> **Pretraining: iSign, 18,000 clips, SimMIM with a 0.75 mask ratio.** Worth **+9.5
> points**, the largest single lever we measured. And the counter-result in the same
> row is the one I would highlight: pretraining *natively* at 16 frames — which should
> have been strictly better, since it removes an interpolation we were doing — **cost
> 10.2 points**, even though it reached a *lower* reconstruction loss. I will come back
> to that on slide 10, because it is the most instructive failure in the project.
>
> **TTA, ensemble and gate:** six views, five members, threshold 0.4. TTA is worth
> +2.5, frame-count diversity across ensemble members helps more than seed diversity
> does, and gating gives 75 percent coverage at 88.7 percent answered accuracy.

### Emphasise
The cache-depth row. It is counterintuitive, it is free at inference time, and it is
the kind of finding that only appears if you are running controlled experiments rather
than tuning.

### Do not say
"We tuned these on the validation set." You did not, and it would be the wrong answer
anyway given slide 8 step 12. These came from controlled one-variable sweeps against a
matched control, each with a second seed.

### If asked
- **"How many of these are single-run results?"** None of the ones we quote as
  interventions. Every claim in that column has a second seed and a matched control,
  and anything under the 2.7-point noise floor is reported as null rather than as a
  gain — the 4,000-epoch result at +0.6 is exactly such a case, and we call it null.
- **"Isn't mixup odd for video?"** We use a mild alpha of 0.2 applied with 50 %
  probability, and we tested the strong alternative and rejected it. At fifteen clips
  per class, some input-space regularisation is necessary; the question is how much.
- **"What is stream dropout?"** We randomly drop one of the three crop streams during
  training, so the model cannot become dependent on, say, always having the face. It
  also matches deployment, where a hand genuinely goes undetected some of the time.

---

# SLIDE 10 — Results and Discussion

### Screen anchor
Two charts. Point at the **top** one and say "this is what we built." Point at the
**bottom** one and say "this is what we found out." They tell opposite stories and the
contrast is the whole talk.

### Script

> This is the results slide, and it has two charts because the project has two results.
>
> **The top chart is the engineering result.** One unchanged 472-clip session-disjoint
> test set, five configurations, cumulative.
>
> We start at **51.7 percent** — 250 epochs, 8 frames. Training properly, 2,000 epochs,
> takes us to **59.7**. Moving to 16 frames drawn from the 32-frame cache: **70.1**.
> Six-view test-time augmentation: **73.3**. A five-member ensemble: **75.8 percent
> top-1, 92.8 percent top-5.**
>
> Now the key inference, in the top-right box, and it is the sentence I would most like
> you to take away from this slide: **the model architecture never changed across that
> entire 24-point improvement.** Same 3.76 million parameters, same eight blocks, same
> width, start to finish. Every point came from training length, frame sampling, cache
> depth, test-time averaging, and ensemble diversity. We spent a significant part of
> this project testing capacity increases, and they were **flat or negative** — tripling
> the token count changed nothing, quadrupling temporal compute produced no coherent
> ordering. The model is data-limited, not capacity-limited.
>
> One detail about that ensemble that is worth the ten seconds. The fifth member we
> added was the **24-frame model, which is the weakest member at 70.3 percent** — and
> adding it still helped. That is the signature of genuine ensemble diversity, members
> making *different* errors, rather than of averaging away noise. Frame count turned
> out to be a better diversity axis than the seed.
>
> **Now the bottom chart, which is the result we did not go looking for.**
>
> Same model, same recipe, same training code. **Only the split file changes.**
>
> A random split by video — the standard protocol — gives **95.7 percent**, which
> reproduces the published INCLUDE range. Grouping near-duplicate takes so the same
> take cannot appear on both sides: **46.5.** Holding out whole recording sessions:
> **31.3 balanced.**
>
> INCLUDE is recorded as back-to-back takes inside shared studio sessions. Under a
> random split, near-duplicate frames of *the same take* land on both sides of the
> train/test boundary. So roughly **65 points of the standard benchmark is
> near-duplicate takes, and a further 7 points is shared recording conditions.**
>
> That is why our headline is 75.8 and not 95. **Both numbers are from this model. Only
> one of them is a measurement of sign recognition.**
>
> Three more things from the boxes on the right.
>
> **The cross-domain warning, and it is the most important limitation in this project.**
> Every number on this slide is measured *inside* INCLUDE. Tested on CISLR — genuinely
> different signers, rooms and cameras — the same checkpoints score **0.3 to 1.5
> percent. Chance is 0.38.** Including the checkpoints scoring 42 percent on INCLUDE.
> We checked whether that was a labelling artefact and it is not: retrieval shows the
> two corpora sign the same gestures ten times better than chance. And 1-NN on the
> features scores the same as the trained head, which locates the failure in the
> **representation**, not the classifier. We also falsified the attractive explanation —
> a learning-rate sweep on the encoder shows in-corpus accuracy tracks encoder training
> monotonically while cross-corpus stays flat at one percent. Finetuning was never
> destroying transferable features. **They were never there.**
>
> **The deployment envelope:** 3.76 million parameters, 0.824 GMACs, 3.70 megabytes
> packed INT8, 12 milliseconds on a desktop CPU. ARM is not yet measured and we do not
> claim it.
>
> **And the confidence gate**, which is how this becomes usable at 75 percent accuracy:
> at threshold 0.4 the system answers 75 percent of clips at **88.7 percent accuracy**,
> and explicitly says "not sure" on the rest. For a communication aid, a declined sign
> costs a repeat; a confident wrong sign costs a wrong sentence.

### Emphasise
The contrast between the two charts, stated explicitly: *"Both numbers are from this
model. Only one of them is a measurement of sign recognition."* Pause after it.

### Do not say
Do not soften the cross-corpus result, and do not let it be discovered by the panel
instead of volunteered by you. Volunteering a fatal-sounding limitation and then
showing you measured its mechanism is the strongest position available to you. Being
caught hiding it is the weakest.

### If asked
- **"So is this system usable?"** On INCLUDE-like video, yes, in the gated regime. On
  arbitrary ISL video from a new signer in a new room, **no**, and nothing we measured
  supports that claim. The fix is data, not architecture — `docs/CAPTURE_PROTOCOL.md`
  is a recording protocol we designed for exactly that, three signers by fifty words by
  three takes.
- **"What failed?"** Three levers, all run against the same control with one variable
  each. Native 16-frame SSL pretraining: **−10.2**. Adding 76 labelled cross-corpus
  clips, 13 % more data: **−5.7**. Doubling epochs again: +0.6, null.
  The second one explains itself through the detection rates — MediaPipe finds a left
  hand in **25.4 %** of CISLR frames against 91.1 % in INCLUDE, so three quarters of
  those left-hand crops are stale boxes showing where a hand recently *was*. We were
  asking the model to transfer between crops of hands and crops of where a hand had
  been. And that revises our own earlier explanation of the cross-corpus failure: we
  had attributed it to a 2.1× sharpness mismatch, which is real and measured, but
  sharpness was never the whole story.
- **"And the pretraining failure?"** That one we find genuinely instructive. The
  16-frame pretrain reached a **lower** reconstruction loss — 0.322 against 0.351 —
  and transferred **ten points worse**. The cause is that our iSign cache holds only
  16 frames, so pretraining at 16 sampled 16 of 16 and had *no temporal jitter at
  all*. Without jitter the frames never move and masked reconstruction has an easy
  shortcut; the loss curve was reporting progress on the shortcut. **The run that
  reconstructed better transferred worse.**
- **"Does the 262-word number improve too?"** Yes. Applying the 16-from-32 recipe to
  the full vocabulary takes session-disjoint top-1 from 22.1 % to **52.8 %**
  (`runs/f16_262w_s0`). We lead with the 50-word number because that is the deployment
  vocabulary and it is where the full ensemble and gating analysis was run.
- **"Which words does it get wrong?"** There is real structure. Of our 50 words, 23
  are above 90 % recall and 9 are below 50 % — and the weak ones are almost all
  **adjectives**: wide, small, fast, hot, old, good, bad, wet. The strong ones are
  nouns and pronouns — shirt, hat, pocket, you, he, she, we. Our reading is that
  adjectives in ISL lean more heavily on non-manual markers and on movement
  *magnitude*, which a 64-pixel crop at 16 frames represents poorly, whereas nouns are
  carried by distinct static handshapes. That is a hypothesis from the per-class recall
  file, not a controlled result.

---

# SLIDE 11 — Dataset, Novelty & IEEE DataPort

### Screen anchor
The red line at the bottom. Address it head-on rather than letting it be spotted.

### Script

> **INCLUDE**, from Sridhar et al., ACM Multimedia 2020. 4,257 unique clips after
> de-duplication, 262 word-level signs across 15 semantic categories, 1920-by-1080 at
> 25 frames per second, two to five seconds per clip, recorded in a single studio in
> Chennai. Per-class support runs from 4 to 22 with a median of 15.
>
> The row that determined this project is the last one: **signer IDs are not shipped
> with the corpus.** There is no field anywhere in the metadata that tells you who is
> signing. That is why a signer-disjoint evaluation is impossible on INCLUDE as
> distributed, and it is why nearly every published number on it is a random split.
>
> **So the novelty, and the first item is the enabling one.** We **reconstruct session
> structure from the camera capture IDs.** Every filename carries an `MVI_` number from
> the camera. We checked the within-class gap distribution and it is cleanly bimodal —
> 2,856 gaps of exactly one, 51 gaps of two, then nothing until a broad hump at 22 to
> 36, which is a later session. Any threshold between 5 and 20 gives an identical
> grouping. That gives **1,334 take-groups**, which cluster globally into **13 session
> blocks.**
>
> On top of that: **four protocols** — official, random-video, take-group and
> session-disjoint — with **automated leakage assertions** that raise rather than warn.
> **Vocabulary tiers** at 30, 50, 100, 137 and 262 words that preserve the
> session-disjoint structure when subsetting, because naively subsetting a split
> reintroduces leakage. And a **cross-corpus CISLR benchmark** over 609 clips covering
> 218 shared words, built with assertions that the test clips appear in neither the
> training rows nor the pretraining pool.
>
> On sourcing: INCLUDE is on **Zenodo**, record 4010759, CC-BY-4.0. The IEEE DataPort
> link on the slide is their ISL keyword index. **I want to be explicit: the corpus we
> report on is not hosted on IEEE DataPort.** We are citing DataPort because the review
> template asks for it; the primary source is Zenodo and that is where our data came
> from. We would rather say that plainly than present a link that implies a provenance
> we do not have.
>
> We also use three further corpora, none for headline results: **CISLR** for
> pretraining and as the cross-corpus test set, **iSign** — 18,000 clips from about
> 11,000 distinct source videos — for scaled pretraining, and **ISL-CSLTR** for
> auxiliary work. Each under its own licence, none redistributed by us.

### Emphasise
Volunteer the DataPort caveat before anyone asks. It is a template requirement you are
answering honestly, and handling it in one confident sentence is much better than being
asked.

### Do not say
"Session-disjoint is equivalent to signer-disjoint." It is a proxy, it is weaker, and
slide 10's cross-corpus result is exactly the evidence that the proxy has limits.

### If asked
- **"How do you know consecutive capture IDs are the same session?"** Two ways. The
  gap histogram is bimodal with a clean empty band, which is what consecutive takes
  followed by a session break looks like and is not what random numbering looks like.
  And a threshold anywhere from 5 to 20 gives the same grouping, so the result is not
  sensitive to the choice. We picked 5 as the conservative end.
- **"Why 13 blocks?"** That is what global clustering across the whole corpus yields.
  And it is a real constraint, not a convenience — with only 13 blocks we cannot make
  validation session-disjoint *as well* without starving it, so validation is carved
  from the training sessions at take-group granularity. Only the test set is
  session-disjoint, because that is the number we report.
- **"Did you have a leakage bug?"** We did, and it is documented in §6.4 of our report.
  An earlier revision quoted 35.8 % for INCLUDE-50 session-disjoint. That split was
  built by clustering session blocks from the *50-word subset* rather than the full
  corpus, so train and test shared 10 of 13 blocks and it was not in fact
  session-disjoint. The number is withdrawn. The builder now takes the whole corpus as
  its clustering reference, every INCLUDE-50 figure here is from the rebuilt split, and
  INCLUDE-263 was never affected because its member set already is the whole corpus.

---

# SLIDE 12 — UI Screens Planned

### Screen anchor
The purple design-principle box on the bottom left.

### Script

> Seven screens, and rather than walk all seven I will give you the principle in the
> purple box and then the three screens that follow from it.
>
> **The principle: confidence gating is part of the interface, not a post-processing
> step.** At threshold 0.4 the model answers about three clips in four; on the rest it
> explicitly says "not sure." A design that hides that and always shows a word would be
> lying to the user 25 percent of the time.
>
> **Screen 1, live capture.** Camera preview with the recognised word and a confidence
> bar — and crucially, **crop health indicators for all three streams**, so the user
> can see when the left hand is not being detected. That is the reliability flag from
> slide 7 surfaced to the person using the device.
>
> **Screen 2, "not sure."** This is the important one. When the gate declines, we do not
> show nothing and we do not show a wrong word. We show the **top-5 shortlist with
> confidences**, which is the metric from slide 6 earning its place — 92.8 percent
> top-5 means the right word is almost always on that list. The user picks, or signs
> again. And their correction becomes labelled data.
>
> **Screen 5, detection diagnostics**, which exists because of slide 10: per-stream
> provenance over the last minute, broken down into direct detection, ROI-rescued,
> interpolated and missing. If the genuine detection rate drops, the system warns the
> user rather than silently degrading — because we know from the CISLR result that
> **detection quality is the strongest predictor we have of whether the model will
> work at all.**
>
> **Screen 6, the vocabulary browser**, shows per-word recall from `per_class.json`.
> Of our 50 words, 23 are above 90 percent recall and 9 are below 50. A user is better
> served by knowing which words are reliable than by a single average.
>
> Screens 3, 4 and 7 are the sentence composer, settings — threshold, vocabulary tier,
> TTA views — and a guided enrolment flow implementing our capture protocol.
>
> One status note: **`islvit/serve.py` is a working local web interface** that runs the
> real model on an uploaded video using the training preprocessing verbatim. The seven
> screens are the designed wearable product; the server is what exists and runs today.

### Emphasise
That the UI is derived from the measurements, not decorated onto them. Screen 2 exists
because of the gating curve, screen 5 because of the cross-corpus finding, screen 6
because of per-class recall. Say that connection out loud.

### Do not say
Do not present the seven screens as built. One is built. Say which.

### If asked
- **"Can you demo it?"** `python -m islvit.serve --run runs/f16_clean_s0` starts a
  standard-library HTTP server — no Flask, no npm — and runs six-view TTA on an
  uploaded clip in well under a second on CPU. It defaults to CPU deliberately, because
  loading a second model onto a GPU mid-training killed a multi-hour run once already.
- **"What happens when the user corrects a prediction?"** Design intent is that the
  correction plus the clip becomes labelled data for that user's signing, which is
  exactly the personalisation the cross-corpus result says we need. Not implemented.

---

# SLIDE 13 — Standard Paper Chosen

### Screen anchor
The title in the red box, and then the **Deployment contrast** row, which is where your
work diverges from it.

### Script

> Our standard paper is **"A comparative analysis of video vision transformers on
> word-level sign language datasets"** — Shawon, Hasan and Mahmud, PLOS ONE, volume 21,
> February 2026.
>
> Five reasons, and the fifth is where we depart from it.
>
> **Same task** — isolated word-level sign classification from video, exactly ours.
> **Same model family** — it compares VideoMAE, ViViT and TimeSformer, and **ViViT's
> factorised-encoder variant is directly the structure we adopted**, so this paper is
> not just related work, it is the design source for slide 7. **Same decision
> variables** — it isolates dataset size, signer appearance, frame distribution and
> frame rate as the factors that determine performance, and every one of those is a
> variable we measured rather than assumed. **Rubric fit** — 2026, SCImago-listed,
> directly in our application domain.
>
> **And the deployment contrast, which is the gap we occupy.** Every backbone in that
> comparison is Kinetics-scale — hundreds of megabytes, designed for server inference.
> ISL-ViT-Tiny targets 3.76 million parameters and 0.82 GMACs. We are asking their
> question under a constraint they do not have.
>
> Their reference values — 96.9 percent on BdSLW60, 81.04 on BdSLW401 — are, note,
> under conventional splits. Our contribution relative to this paper is to take their
> architectural finding, compress it by orders of magnitude, and then evaluate it under
> a protocol that their numbers were not subject to.

### Emphasise
That the paper is the *design source* for the factorisation, not merely a citation.
That is a much stronger relationship to a standard paper than "it is in the same area."

### Do not say
"We outperform this paper." Different datasets, different splits, different model
scale. Not comparable, and claiming it would undercut the whole protocol argument you
just spent slide 10 making.

### If asked
- **"Why not VideoMAE or TimeSformer instead?"** TimeSformer's divided attention still
  attends across all frames at each spatial location, so its cost scales with clip
  length more aggressively than a factorised encoder. VideoMAE is a pretraining method
  and needs large-scale video pretraining data we do not have for this domain. ViViT's
  factorised encoder gives the cleanest cost separation — and a natural place to inject
  our geometry tokens between the two stages.

---

# SLIDE 14 — Similar Products / Systems

### Screen anchor
The **Limitation / gap** column, right-hand side.

### Script

> Five closest systems, and the column to read is the right-hand one.
>
> **TinyMSLR** is closest on deployment: hybrid ConvNeXt plus Swin with knowledge
> distillation, under 2.7 million parameters, edge latency measured. Its gap is
> evaluation — twenty controlled classes, no signer-disjoint split.
>
> **Dynamic Kannada SLR** is the strongest edge result in the set: 1.1 megabytes, 16.2
> milliseconds on a phone, an Indian sign language. But it consumes **landmarks, not
> RGB** — which means it inherits every failure of the landmark extractor and can never
> see anything the extractor does not encode. Our slide-10 finding that CISLR
> detection rates collapse to 25 percent is precisely the failure mode that approach is
> exposed to and cannot detect.
>
> **The ISL e-governance agent** is the closest ISL system: hybrid CNN plus ViT,
> real-time, 38,539 clips. And it reports its own gap for us — **a seven-point drop
> between known and unknown signers**, with no signer-disjoint benchmark. That is the
> same effect we measured, at a much larger corpus scale.
>
> **SignViT** reaches 99 percent plus, on static images — no temporal modelling at all,
> and no edge budget. **The video ViT comparison** is our standard paper: right family,
> too large.
>
> The pattern across all five: the edge systems compress but do not test signer
> generalisation, and the accurate systems test on splits that permit memorisation.
> **We are the intersection, and the cost of being in the intersection is that our
> headline number is 75 instead of 95.**

### Emphasise
The closing line. It reframes a weaker-looking number as the *consequence* of a
methodological choice, which is exactly what it is.

### Do not say
Do not disparage these systems. Each is stronger than yours on its own axis, and saying
so costs nothing and buys credibility.

### If asked
- **"Is 75 % good?"** Against these systems' headline numbers, no. Against these
  systems' numbers *under our protocol*, we do not know, because none of them publishes
  one — and the one paper in our survey that did run the comparison, "Beyond Perfect
  Scores," found drops of 15 to 83 points. Our own drop was 64. The honest answer is
  that 75 % session-disjoint and 95 % random-split are not the same kind of number, and
  ours is the harder kind.

---

# SLIDE 15 — Product Abstract, State of the Art, Novelty & Research Gap

### Screen anchor
The bold line at the bottom of the novelty box: *"a compact ViT evaluated under a
protocol that does not permit memorisation; not a wholly new transformer family."*

### Script

> This slide is our claim, stated as narrowly as we can defend it.
>
> **Abstract.** ISL-ViT-Tiny recognises isolated ISL words from short RGB clips using a
> compact factorised spatio-temporal Vision Transformer, designed for offline wearable
> inference.
>
> **State of the art.** Video ViTs show strong word-level sign recognition; edge systems
> prove transformer inference can be compact. The combination remains incomplete.
>
> **Research gap.** No surveyed work combines an RGB Vision Transformer, Indian Sign
> Language, wearable-scale deployment and leakage-controlled session-disjoint
> evaluation in one system.
>
> **Five novelty claims**, and I will give the mechanism for each rather than just the
> name.
>
> **One, geometry re-injection with a zero-initialised projection.** Cropping deletes
> sign location, which is linguistically defining. We project the box coordinates back
> in through a branch that contributes exactly zero at initialisation, so it refines
> pretrained appearance features rather than swamping them.
>
> **Two, a per-token reliability flag from detector provenance.** The model is told
> which crops came from a genuine detection and which are carried-forward boxes. It is
> not asked to infer that from pixels.
>
> **Three, session and take reconstruction from camera capture IDs.** This is the one
> that enables everything on slide 10, and it recovers structure from a corpus that
> ships no signer identity.
>
> **Four, cache depth as an inference diversity axis.** Storing more frames than you
> feed improves accuracy at identical inference cost, and the gain from test-time
> averaging tracks the ratio monotonically — we measured +3.8, +3.4, +0.6, +0.2 across
> four headroom levels.
>
> **Five, confidence gating as a delivered interface**, not as a reported metric.
>
> **And then the honest claim, in bold, which we put on the slide deliberately: this is
> a compact ViT evaluated under a protocol that does not permit memorisation. It is
> not a wholly new transformer family.** The blocks are standard pre-norm transformer
> blocks. What is new is the conditioning, the protocol, and the fact that every
> intervention was measured against a noise floor.

### Emphasise
The bold honest-claim line. Read it verbatim from the slide. A panel that has sat
through overclaimed novelty all day will notice.

### Do not say
Do not upgrade any of the five. Especially not "we invented factorised attention" —
ViViT did, and slide 13 says so.

### If asked
- **"Which of the five is most defensible?"** Three, the session reconstruction, because
  it is the one that makes every other number in the talk meaningful, and it generalises
  to any corpus with sequential capture metadata — which is most of them.
- **"Would geometry re-injection help other tasks?"** Anywhere a crop-based pipeline
  discards spatial context that matters — gesture recognition, pose-conditioned action
  recognition, egocentric hand tasks. The zero-init trick is more general still: it is
  how you add a low-dimensional side channel to a pretrained model without disturbing
  it early.

---

# SLIDE 16 — Individual Contributions

### Screen anchor
The pink box at the bottom: *data integrity → model → evaluation → deployment.*

### Script

> Three technically distinct contributions that meet at one pipeline, and each maps to
> named files in the repository with commit history behind them.
>
> **Dhrusheek** owns **protocol and evaluation**: reconstructing take-groups and
> session blocks from the capture IDs, building the four protocols, the leakage
> assertions, and then the evaluation machinery on top — test-time augmentation,
> confidence gating, and the five-seed noise-floor study. `splits.py`, `tta.py`,
> `gate.py`. That work is why every number in this deck carries a protocol name.
>
> **Kushala** owns **the model**: the factorised spatio-temporal ViT, the geometry
> re-injection with its zero-initialised projection, the missing-token flag, DeiT-Tiny
> weight loading with position-embedding resizing, the training loop, and the EMA warmup
> fix. `isl_vit.py`, `train.py`, and the hyperparameter recipe on slide 9.
>
> **Kushal** owns **data and deployment**: the two-stage MediaPipe crop pipeline with
> per-crop provenance, the 32-frame re-extraction that lifted the ceiling on slide 10,
> the iSign self-supervised pretraining, INT8 export with measured rather than computed
> model size, and the serving path. `crops.py`, `pretrain.py`, `export.py`, `serve.py`.
>
> The line at the bottom is the honest description of how this worked: the
> contributions are separable but not independent. **Data integrity feeds the model,
> the model feeds evaluation, evaluation feeds deployment** — and the important
> findings in this project all came from the joins. The cache-depth result needed
> Kushal's re-extraction and Dhrusheek's TTA to be visible at all. The cross-corpus
> failure was only explicable once the detection rates from the data side were read
> against the evaluation numbers.

### Emphasise
That the findings live at the joins. It is true here, and it answers the unspoken
question of whether three people did one person's work in parallel.

### Do not say
Do not claim exclusive ownership of shared files. If asked who wrote a specific
function, say who led it and who reviewed it.

### If asked
- **"Can each of you answer for the others' parts?"** Yes — and that is the real test of
  this slide, so be ready. Each of you should be able to explain the factorisation, the
  two-stage rescue, and the session reconstruction at slide depth regardless of whose
  name is on them.

---

# SLIDE 17 — Denoising / Noise-Robustness Approach

### Screen anchor
The grey subtitle. Read it before anything else on the slide.

### Script

> Two scoping sentences first, because I do not want this slide misread.
>
> **This is a template-required section and it is a planned extension. It is not used
> in the 51.7 to 75.8 results on slide 10.** Nothing on this slide or the next one
> contributed a single point to any number we have shown you.
>
> With that said, it is not an arbitrary addition, because the noise types are chosen
> from failures we actually observed in our own pipeline.
>
> **Motion blur** is first, because it is the measured cause of our detector losing
> roughly 19 percent of frames at full resolution — hands move fast at 25 frames per
> second. **Compression and resampling artefacts**, because our crops are upscaled or
> downscaled from native resolution, and we measured a **2.1× sharpness difference**
> between CISLR and INCLUDE crops that demonstrably damaged transfer. **Illumination
> and contrast**, because session variation is the effect that slide 10 is about.
> Plus Gaussian sensor noise for low light, and mild crop misalignment.
>
> The design is the standard restoration setup: take a clean crop from the 128-pixel
> cache, corrupt it synthetically, train a denoiser to predict the residual, and feed
> the cleaned crop to the ViT. The training pair is the noisy crop and the original, so
> the supervision is free — we already have the clean crops.
>
> **And the constraint in bold at the bottom is the reason we have not simply bolted
> this on.** Denoising has to preserve hand boundaries and sign-location cues.
> Excessive smoothing would remove exactly the discriminative detail the model depends
> on — a 64-pixel hand crop has very little detail to spare. So this needs to be
> evaluated as *downstream classification accuracy*, not as PSNR on the reconstruction,
> and the honest outcome is that it might well hurt. We would rather say that now than
> report a PSNR number that means nothing for our task.

### Emphasise
Twice: it is planned, and it contributed zero points. Then the bold constraint, because
it shows you have thought about whether it would actually help rather than assuming.

### Do not say
Do not imply any reported accuracy includes denoising. If a panelist comes away
believing that, your slide-10 numbers become unauditable.

### If asked
- **"Why not just add noise as augmentation instead?"** Reasonable, and cheaper, and we
  partially do — colour jitter, grayscale and resolution jitter are in the recipe. But
  augmentation teaches the model invariance to noise it has seen; a denoiser is a
  separate reusable front end that helps every downstream model. Also, our
  strong-augmentation experiment on slide 9 was a **null with worse calibration**, so
  we are not assuming more augmentation is free.
- **"Would you evaluate it on real noise or synthetic?"** Synthetic to train, real to
  evaluate — and CISLR is the natural real test set, since we have measured exactly how
  it differs.

---

# SLIDE 18 — Deep Learning Model for Denoising: DnCNN

### Screen anchor
The bottom grey line — the scoping statement — and then the architecture strip.

### Script

> The specific model, again template-required, and again: **proposed, not a component
> of the reported results.**
>
> **DnCNN**, Zhang et al., IEEE Transactions on Image Processing 2017 — "Beyond a
> Gaussian Denoiser." Architecture is a first Conv 3-by-3 plus ReLU producing 64
> features, then fifteen Conv-BatchNorm-ReLU blocks, and a final layer predicting the
> **residual** — the noise — which is subtracted from the input to give the clean
> image.
>
> Four reasons it fits here.
>
> **Residual learning directly models the noise rather than the image**, which is the
> right target when the signal is mostly preserved and the corruption is additive —
> easier to learn, and it degrades gracefully toward the identity when there is nothing
> to remove. That last property matters for us: on a clean crop, a residual denoiser
> that predicts near-zero does almost nothing, whereas a direct-reconstruction model
> would still pass the crop through a lossy bottleneck.
>
> **It is fully convolutional**, so it runs at any input size — we could apply it at
> 128 pixels on the cache or at 64 on the model input.
>
> **It is lightweight relative to our ViT**, which matters given a 3.7-megabyte total
> budget. Though I will flag honestly that a 17-layer, 64-feature DnCNN applied to
> three crops at sixteen timesteps is **not obviously cheap relative to a 0.82-GMAC
> model**, and sizing that is part of the work, not an assumption.
>
> **And it inserts cleanly** before the 64-pixel input stage, without touching anything
> on slide 7.
>
> Reference and implementation are on the slide. And the final line, which I will read
> as written: **DnCNN is a proposed denoising component for future robustness
> experiments, not a component of the current reported ISL-ViT-Tiny results.**
>
> That is our presentation. To close on the one sentence: we built a 3.76-million
> parameter sign-language transformer that runs in 12 milliseconds and 3.7 megabytes —
> and the more useful thing we found is that **most of what the standard benchmark
> measures is not sign recognition.** Our honest number is 75.8 percent, we know
> exactly which 24 of those points we added and that none of them were architectural,
> and we know precisely where it stops working. Thank you — we are happy to take
> questions.

### Emphasise
The closing three-part summary: what you built, what you found, what you know the limit
is. Rehearse it as a unit; it is the last thing the panel hears.

### Do not say
Do not end on DnCNN. It is the weakest slide in the deck because it is the least yours.
Take thirty seconds to close on your own result.

### If asked
- **"Have you implemented DnCNN?"** No. It is specified and justified, not built. Our
  next robustness step would actually be the capture protocol in
  `docs/CAPTURE_PROTOCOL.md` — new signers address the failure we *measured*, whereas
  denoising addresses one we have hypothesised.

---

# Appendix A — Cross-cutting questions

Questions that are not tied to one slide. Each answer is grounded in a repository
artefact.

**"What is the single most important result?"**
That roughly 65 points of the standard INCLUDE benchmark is attributable to
near-duplicate takes and a further 7 to shared recording conditions — same model, same
recipe, only the split file changes. Everything else in the project follows from it.

**"What would you do with three more months?"**
Record new data, in this order. First, the capture protocol in
`docs/CAPTURE_PROTOCOL.md` — three signers by fifty words by three takes, about 450
clips — because the cross-corpus result says signer diversity is the binding
constraint and we falsified the alternatives. Second, re-extract iSign at 32 frames to
fix the pretraining-jitter problem; we know the mechanism, we are blocked on 80 GB of
disk. Third, measure ARM latency, because 12 milliseconds is a desktop CPU number and
the wearable claim is not complete without it.

**"What is the weakest part of this work?"**
Cross-corpus generalisation, and we would not argue. The same checkpoints score at
chance on a different corpus. We have located the failure in the representation rather
than the head, we have excluded the labelling and encoder-erasure explanations by
measurement, and we have a plan — but we have not fixed it.

**"How much compute did this take?"**
One RTX 3060 with 12 GB, 32 GB of RAM, Windows 11. 81 recorded runs. Crop extraction
for the 32-frame cache was 98 minutes at six workers with zero failures. Everything in
this project is reproducible on one consumer GPU, which was a deliberate constraint.

**"Did you get anything wrong?"**
Three claims, each made against a weak reference or too few seeds and each overturned
by a properly controlled follow-up — including one case where we reversed a previous
correction. They are documented in the report rather than edited out. We also recorded
two predictions in advance and both were wrong: that 32 frames would underperform 16,
and that native 16-frame pretraining would be our best lever. The common failure was
reasoning from a mechanism that was real but incomplete. The procedural response is
that every claim now gets a second seed and a matched control before it is written
down.

**"Why is the GitHub README showing 68.9 % when your slide says 75.8 %?"**
Because the README predates §11.11 of the report. 68.9 % was the best two-seed
ensemble before the 32-frame re-extraction; 75.8 % is the current five-member ensemble
on the same unchanged 472-clip test set. The report's executive summary and §11.11 are
current; the README is stale and is being updated. **Fix this before the review if you
can** — see Appendix B.

**"What is the difference between 262 and 263 classes?"**
262 is the correct class count; run tags in the repo use `263` for historical reasons
from an early off-by-one in a category count. Say 262.

---

# Appendix B — Pre-review checklist

Things to do to the repository before the panel opens it.

| # | Item | Why |
|---|---|---|
| 1 | Update `README.md` headline from 68.9 % / 3.80 M / ≈3.8 MB to 75.8 % / 3.76 M / 3.70 MB | It is the first thing a panelist opens and it currently contradicts slide 10 |
| 2 | Decide the gating number and use one everywhere | Slide 4 says 80.7 % at 76 % coverage (the older two-seed figure), slide 6 and slide 10 say 88.7 % at 75.2 % (`runs/gate_mixed.json`), the report §11.11 says 90.7 % at 75 % (five-member). All three are real, from different ensembles. **Quote 88.7 % at 75 % coverage** — it is the one on the most slides and it traces to a committed file |
| 3 | Commit the untracked results | `runs/baseline_random_s0/summary.json`, `runs/f16_262w_s1/`, `runs/f16_clean_s0/per_class.json` are uncommitted. Slide 12's vocabulary browser and Appendix A's per-class answer both depend on `per_class.json` |
| 4 | Reconcile the asset counts | Slide 5 says 79 runs / 21 splits / 26 scripts; the repo has 81 run directories, 21 splits, 13 configs, 25 `run_*.sh`. Either update the slide or say "about 80" |
| 5 | Have `islvit/serve.py` running before you walk in | If the panel asks for a demo, starting it cold costs you two minutes of silence |
| 6 | Print one `summary.json` | Have `runs/f16_clean_s0/summary.json` open in a tab. "Every number traces to a file" is much stronger when you can show the file in four seconds |
| 7 | **Fix or drop the 48× on slide 7** | The slide says ≈48×, the report §2.3 table says ≈0.02× (= 50×), and the report's own formula `O(P²)+O((T·S)²)` gives ≈180×. Counting the spatial encoder the 24 times it actually runs gives ~22×. Four different numbers for one claim. Recompute it, state the accounting, or replace the multiplier with the asymptotic statement |
| 8 | Note slide 7's two configurations | The diagram is T=16 (49 temporal tokens); the cost box is T=8 (25 temporal tokens, 408 naive). Both are correct but they are different configs on one slide — say which you are quoting |

---

# Appendix C — Number card

Memorise these. Everything else can be read off a slide.

| | |
|---|---|
| Parameters | 3.76 M (50-word) · 3.80 M (262-word) |
| Compute | 0.824 GMAC / clip |
| Size | 3.70 MB packed INT8 (measured) |
| Latency | 12 ms / clip, desktop CPU |
| Input | 16 frames × 3 streams × 64×64, from a 32-frame / 128 px cache |
| Corpus | INCLUDE — 4,257 clips · 262 signs · median 15 clips/class |
| Structure recovered | 1,334 take-groups → 13 session blocks |
| Test set | 472 clips, INCLUDE-50, session-disjoint |
| Trajectory | 51.7 → 59.7 → 70.1 → 73.3 → **75.8 %** top-1 · 92.8 % top-5 |
| Leakage ladder | random 95.7 → take-group 46.5 → session-disjoint 31.3 balanced |
| Leakage decomposition | ~65 pts near-duplicate takes · ~7 pts shared sessions |
| Gating | τ = 0.4 → 75 % coverage at 88.7 % accuracy, 66.7 % yield |
| Noise floor | sd 1.9 per run; ~2.7 pts for a difference of two runs |
| Biggest levers | iSign pretraining +9.5 · 250→2000 epochs +7.9 · more INCLUDE data +7.7 · 8→16 frames +6.4 · cache depth +3.2 · TTA +2.5 |
| Rejected | native-16f SSL −10.2 · +76 CISLR clips −5.7 · 4,000 epochs +0.6 (null) · strong augmentation (null, worse calibration) · grokking (negative) |
| Detection | INCLUDE L-hand 91.1 % · CISLR L-hand 25.4 % |
| Cross-corpus | 0.3–1.5 % on CISLR; chance 0.38 % |
| 262-word strict | 22.1 % → 52.8 % with the 16-from-32 recipe |

---

# Appendix D — Glossary

Every term that appears in the deck, defined generally and then grounded in what it
actually is in this project. If a panelist points at a word on a slide, this is the
answer.

## D.1 Evaluation and protocol

**Leakage** — any situation where information about the test set reaches the model
during training, making the test score an overestimate. It is not always a bug in the
code; here it is a property of how the corpus was *recorded*, and the split has to
defend against it.

**Disjoint** — two sets share no elements. Every split has disjoint *clips*; the
question is what else is disjoint.

**Take** — one recording of one sign. INCLUDE records several back-to-back takes of
the same word, seconds apart, so consecutive takes are near-duplicates.

**Take-group** — the equivalence class of takes belonging to one recording run. Built
in `islvit/data/splits.py` by grouping clips of the same class whose `MVI_` capture IDs
are within `TAKE_GAP_THRESHOLD = 5`. 1,334 of them.

**Session / session block** — one recording sitting: same signer, room, clothing,
lighting. Recovered by clustering capture IDs *corpus-wide* rather than per-class.
13 of them. Clustering per-class instead is the bug documented in §6.4 of the report.

**Protocol** — the rule for assigning clips to train/val/test. Four exist here:
`official`, `random-video`, `take-group`, `session-disjoint`. **The model and recipe
are identical across all four; only the split file changes.**

**Session-disjoint** — the strict protocol. Whole session blocks are held out for
test, so no session appears on both sides. This is the number this project reports.

**Signer-disjoint** — no signer appears in both train and test. **Impossible on
INCLUDE**, which ships no signer identity. Session-disjoint is the proxy, and it is
strictly weaker.

**Leakage assertion** — an automated check that raises (not warns) if a session or
take-group straddles the train/test boundary. Runs every time a split is written.

**Cross-corpus evaluation** — testing on a genuinely different dataset. Here: 609
CISLR clips covering 218 shared words, different signers/rooms/cameras. The result
(0.3–1.5 %, chance is 0.38 %) is the binding limitation of the project.

**Leakage ladder** — the bottom chart on slide 10. The same model scored under
progressively stricter protocols: 95.7 → 46.5 → 31.3.

**Vocabulary tier** — a subset of the 262 words (30/50/100/137/262) built so that
subsetting *preserves* session-disjoint structure. Naively subsetting a split
reintroduces leakage.

## D.2 Metrics

**Top-1 accuracy** — fraction of clips where the highest-scoring class is correct.

**Top-5 accuracy** — fraction where the correct class is in the top five. Not padding:
it is what the UI shortlist on slide 12 shows. 92.8 % here.

**Balanced accuracy** — the mean of per-class recalls, so every word counts equally
regardless of how many test clips it has. On INCLUDE-50 session-disjoint it reads
31.3 % where top-1 reads 14.3 % **for the same predictions**, because the test set is
adversarially imbalanced (corr(test count, train count) = −0.73).

**Recall (per class)** — of the clips that truly are word *w*, the fraction predicted
*w*. Written per word to `per_class.json`.

**Precision** — of the clips predicted *w*, the fraction that truly are *w*.

**Macro F1** — unweighted mean over classes of the harmonic mean of precision and
recall, 2PR/(P+R). Exposes per-class failure structure that an average hides.

**Coverage** — fraction of clips the gated system chooses to answer at all.

**Selective accuracy** — accuracy *among the answered clips only*. 88.7 % at tau = 0.4.

**Yield** — coverage x selective accuracy: the fraction of all clips that are both
answered and correct. 66.7 %. This is what the user actually experiences.

**Chance level** — 1/n_classes. 2 % at 50 words, 0.38 % at 262. Quoted so
cross-corpus numbers can be read against something.

**Seed noise floor** — how much a score moves from the random seed alone, with
everything else fixed. Measured over five seeds: sd 1.9 points per run, so the sd of a
*difference between two runs* is about 2.7. **Any single-run comparison below ~5 points
in this project is unresolved, not a result.**

**Matched control** — the run an intervention is compared against, identical in every
respect except the one variable. Every claim in the deck has one.

## D.3 Architecture

**ViT (Vision Transformer)** — a transformer applied to images by cutting them into
fixed patches and treating each patch as a token.

**Patch embedding** — the layer that turns patches into tokens. Here a `Conv2d(3->192,
kernel=16, stride=16)`; on a 64 px crop that gives a 4x4 = 16-patch grid.

**Token** — one vector in the sequence the transformer attends over. Width 192 here.

**CLS token** — a learned vector prepended to the sequence whose *output* is used as
the summary of the whole sequence. Both stages have one.

**Positional embedding** — learned vectors added to tokens so the model knows where
each patch sat, since attention is permutation-invariant. Loaded from DeiT-Tiny and
**bicubically resized** from its 14x14 grid to our 4x4.

**Self-attention** — each token computes a weighted sum over all tokens, weights from
softmax(QK^T/sqrt(d)). Cost is quadratic in sequence length, which is the whole problem.

**Multi-head attention (MHSA)** — attention run in parallel in several subspaces and
concatenated. 3 heads x head_dim 64 = 192 here.

**Factorised (spatio-temporal) encoder** — the ViViT variant this model adopts:
attend *within* a crop first (Stage A, 17 tokens), then *across* crops (Stage B, 25
tokens), instead of jointly over all 408. Turns O(408^2) into O(17^2) + O(25^2), about
**1/48 the attention cost** — the reason the model fits an edge budget.

**Stage A / spatial encoder** — 4 blocks over the patches of one crop. **Weights are
shared across all 48 crops**, which is why 94 % of the compute sits here but the
parameters do not multiply.

**Stage B / temporal encoder** — 4 blocks over the 48 per-crop embeddings, attending
across time *and* stream jointly.

**Stream** — one of the three crops per timestep: left hand, right hand, face.

**Transformer block** — LayerNorm -> attention -> residual add -> LayerNorm -> MLP ->
residual add.

**Pre-norm** — normalisation *before* the sublayer rather than after
(`x + attn(norm(x))`). More stable for deep stacks.

**LayerNorm** — normalises each token across its features. eps 1e-6.

**MLP ratio** — the expansion inside the block's feed-forward layer. Ratio 4 gives
192 -> 768 -> 192.

**GELU** — the smooth activation used in the MLP and the geometry projection.

**Residual / skip connection** — adding a sublayer's input to its output, so gradients
have a direct path.

**DropPath (stochastic depth)** — randomly drop an entire residual branch for a sample
during training, so the network trains as an implicit ensemble of depths. Rate ramps
linearly 0 -> 0.1 across the blocks.

**DeiT-Tiny** — the ImageNet-pretrained ViT whose geometry (dim 192, 3 heads) the
spatial encoder copies **exactly**, so its weights load with no projection layer.
Width is *pinned* to 192 for this reason, and this initialisation is the single
biggest lever against 15 clips per class.

**Geometry re-injection** — cropping to the hand deletes *where the hand is*, which is
a defining linguistic parameter of a sign. The box's (centre x, centre y, size) is
projected back in via `Linear(3->192) -> GELU -> Linear(192->192)` and added to the token.

**Zero-initialisation** — the last layer of that geometry projection starts at exactly
zero, so at step 0 the branch contributes nothing and the model begins as a pure
appearance model, learning to use location *gradually*. Prevents three raw
coordinates from swamping a 192-d pretrained feature when gradients are largest.

**Missing embedding / reliability flag** — a learned vector added to any token whose
crop came from a carried-forward box rather than a genuine detection. Telling the
model "this is absent" beats making it infer absence from all-zero pixels — and in
ISL the absence of a second hand is itself informative.

**Head** — the final `Linear(192 -> n_classes)`.

**Logits** — the raw pre-softmax scores out of the head.

**Softmax** — turns logits into a probability distribution over classes.

## D.4 Training

**Epoch** — one pass over the training set. 2,000 here.

**Cross-entropy loss** — the standard classification objective, minus sum of q log p.

**Label smoothing** — replace the hard 1/0 target with 0.9 and 0.1 spread over the
rest, epsilon = 0.1. Discourages overconfidence, which matters because confidence is a
*product feature* here (the gate).

**AdamW** — Adam with *decoupled* weight decay. Applied at 0.05, and **only to tensors
with ndim > 1** — weights get decay, norms and biases do not.

**Learning rate** — 5e-4 base.

**Warmup** — ramping LR up from near zero over the first 10-15 epochs. Training was
unstable without it.

**Cosine schedule** — LR decays along a cosine curve from base to near zero over
training.

**`backbone_lr_scale`** — the spatial encoder trains at 0.1 x the base LR: the
pretrained encoder is *refined*, not erased. Swept 0.0 / 0.01 / 0.1 giving 31.8 / 38.7 /
41.8 %, so neither freezing nor full-rate training is right.

**Gradient clipping** — cap gradient norm at 1.0 to prevent a single bad batch from
destabilising training.

**EMA (exponential moving average)** — keep a slowly-updated running average of the
weights and evaluate *that*. Implemented in `islvit/train.py` as
`ema <- delta*ema + (1-delta)*weights`.

**EMA decay warmup** — delta_t = min(0.99, (1+t)/(10+t)). A **fixed** 0.999 has a
~1,000-step time constant, and a 50-word run is only ~1k updates — so the average
would have been dominated by the *initialisation* for essentially the whole run. This
was a real bug, and the ramp is the fix.

**bf16 (bfloat16)** — 16-bit float with FP32's exponent range. Halves memory, so batch
64 fits in 12 GB.

**Checkpoint selection** — which epoch's weights you keep. Here: **the final EMA
weights, not best-validation**, because val is carved from the *training* sessions and
so is not session-disjoint from training — it is not a clean selection signal.

**Augmentation** — random transformations applied during training so the model learns
invariances. Here: temporal jitter, random crop 128->64, flip-with-stream-swap, plus or
minus 0.4 colour, grayscale 0.15, stream dropout 0.15, mixup alpha 0.2.

**Temporal jitter** — sampling a random frame inside each of the 16 uniform segments
rather than the centre. **Needs cache headroom to exist at all** — this is the
mechanism behind the whole 32-frame result.

**Headroom** — cached frames divided by sampled frames. 32-from-32 is 1.0x (no jitter
possible); 16-from-32 is 2.0x. TTA gain tracks it monotonically: +3.8, +3.4, +0.6, +0.2.

**Flip-with-swap** — mirroring the image **and** exchanging the L/R hand streams and
mirroring the box geometry. Mirroring pixels alone would be physically incoherent.

**Stream dropout** — randomly drop one of the three crops so the model cannot depend on
always having, say, the face. Also matches deployment, where hands genuinely go
undetected.

**Mixup** — train on a convex blend of two clips and their labels, alpha = 0.2, applied
50 % of the time. Mild input-space regularisation for a 15-clip-per-class problem.

**Cutmix / random erasing** — stronger augmentations, **tested and rejected**: top-1
null at both frame counts and measurably worse calibration.

**Calibration** — whether predicted confidence matches actual correctness. Critical
here because tau = 0.4 is a confidence threshold; a miscalibrated model breaks the gate.

**Transfer learning** — initialising from weights trained on another task.

**Self-supervised pretraining (SSL)** — learning from unlabelled video by solving a
pretext task. Worth **+9.5 points**, the largest single lever measured.

**SimMIM / masked image modelling** — mask 75 % of patches, **replace them with a
learned mask token**, and reconstruct the missing pixels. SimMIM-style (mask tokens go
through the encoder) rather than MAE-style (masked patches are dropped from the
encoder input entirely).

**Reconstruction loss** — the pretraining objective. **And a warning from section
11.11: the run that reconstructed *better* (0.322 vs 0.351) transferred *10 points
worse*.** Without jitter, masked reconstruction has an easy shortcut and the loss curve
reports progress on the shortcut. Reconstruction loss is not a proxy for transfer
quality.

**Grokking** — delayed generalisation, where test accuracy jumps long after training
loss has plateaued. **Tested directly, negative in both weight-decay arms** — but the
test revealed every run in the project was undertrained (250 -> 2,000 epochs = +7.9).

**Knowledge distillation** — training a small student to match a larger teacher's
outputs. `distill.py` exists; **no reported number depends on it.**

**Ablation** — removing or varying one component to measure its contribution.

## D.5 Data pipeline

**MediaPipe Holistic** — Google's pose + face + hand landmark model, run on the full
frame. Finds a hand in only ~79 % of frames here (190 px hand in a 1920 px frame,
plus motion blur).

**HandLandmarker** — MediaPipe's dedicated hand model, run in stage 2 on an upscaled
crop.

**Two-stage rescue** — when full-frame detection misses, take an ROI from the pose
wrist/elbow (side = 2.8 x forearm), upscale to 256 px, re-run HandLandmarker.
**Recovers 10-15 % of frames** that stage 1 alone would have lost.

**ROI (region of interest)** — the sub-rectangle passed to stage 2.

**Provenance / box source** — a per-crop code recording *how* that box was obtained:
direct (81.4 % L-hand), ROI-rescued (9.9 %), interpolated (8.6 %), missing (0.1 %).
Added after an earlier version reported 100 % coverage while producing crops of shirt
fabric and studio wall — **coverage measured box existence, not box correctness.**

**Interpolated box** — the nearest real detection carried forward. Flagged unreliable
to the model, never presented as a detection.

**Detection coverage** — share of frames from genuine detections. The strongest
predictor of cross-corpus transfer found in this project: INCLUDE L-hand **91.1 %**
against CISLR **25.4 %**.

**Crop cache / memmap** — crops written once to `crops.npy`, a uint8 memory-mapped
array indexed lazily from disk. Makes augmentation free at train time: no video
decode, no MediaPipe in the loop. `cache128_f32` is 18.7 GB, 4,257 clips, 98 min,
zero failures.

**De-duplication** — removing repeated clips from the corpus. 4,257 unique remain.

**iSign / CISLR / ISL-CSLTR** — the auxiliary corpora: iSign (18k clips) for scaled
pretraining, CISLR for pretraining and as the cross-corpus test set, ISL-CSLTR for
auxiliary work.

## D.6 Inference and deployment

**TTA (test-time augmentation)** — average predictions over several transformed views
of the same clip. 3 phases x 2 flips = **6 views**. **Probabilities averaged, not
logits** — the views are alternative observations, so a confidently-right view should
outvote an undecided one, and averaging logits lets one large negative dominate.
Nothing is trained; it cannot overfit; it also cannot fix a model wrong in every phase.

**View** — one deterministic transformation of a clip at test time.

**Ensemble** — average probabilities across several trained models. 5 members here.
**Frame count turned out to be a better diversity axis than the seed** — the 24-frame
model helped despite being the *weakest* member, which is the signature of genuine
diversity rather than noise-averaging.

**Confidence gating** — emit a word only if max probability is at least tau, else emit
"not sure." tau = 0.4 gives 75 % coverage at 88.7 % accuracy. **Note: tau was chosen on
the test set**, so it is a characteristic curve, not a validated operating point.

**Abstention** — the "not sure" output. A first-class result, not an error: a declined
sign costs the user a repeat, a confident wrong sign costs them a wrong sentence.

**GMAC** — giga multiply-accumulate operations, the standard compute measure. 0.824
per clip, about 94 % of it in Stage A.

**Parameters** — learned weights. 3,760,754 at 50 words; 94.6 % sit in the 8 blocks.

**Quantisation** — storing weights at lower precision. **INT8** = 8-bit integers.

**Dynamic quantisation** — `quantize_dynamic` converts `nn.Linear` **only** — 95.7 %
of parameters here. The Conv2d patch embed, LayerNorms and position/time embeddings
stay FP32 at 4 bytes each, and quantised tensors carry per-channel scales and
zero-points. **So 3.76 M parameters is not a 3.76 MB file** — the measured packed size
is 3.70 MB, weighed on disk by `islvit/export.py`, not computed.

**Latency** — 12 ms per clip, single-clip, desktop CPU. **ARM is not measured** and is
not claimed.

**Sliding window** — 0.8 s window, 0.25 s stride over the live camera stream.

**Train/serve skew** — when deployment preprocessing differs from training
preprocessing, so a model that scores well offline fails on real input. Removed
*structurally* here: `serve.py` imports `extract` and `views` from `predict.py` — the
same functions that built the cache.

## D.7 Denoising slides (17-18)

**Denoising** — recovering a clean image from a corrupted one. **Planned extension;
contributes zero points to any reported result.**

**DnCNN** — Zhang et al., IEEE TIP 2017. Conv 3x3 + ReLU (64 features), then 15
Conv-BN-ReLU blocks, predicting the noise residual.

**Residual learning (denoising sense)** — predict the *noise* r(x) and subtract:
clean = x - r(x). Easier to learn than the image, and it degrades gracefully toward
the identity on an already-clean crop.

**Batch normalisation** — normalises activations across the batch; standard in DnCNN,
not used in the ViT (which uses LayerNorm).

**Fully convolutional** — no fixed input size, so it could run at 128 px on the cache
or 64 px at the model input.

**PSNR** — peak signal-to-noise ratio, the usual denoising metric. **Deliberately not
our success criterion** — this must be judged on downstream classification accuracy,
because over-smoothing would destroy exactly the hand-boundary detail a 64 px crop
cannot spare.
