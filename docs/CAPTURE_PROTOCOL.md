# Recording Protocol — Signer-Independent Validation Set

## Why this exists

Every accuracy figure in the model report is measured on **one studio, one city,
one recording setup**. INCLUDE ships no signer identifiers at all, so the two
strict protocols — `take-group` and `session-disjoint` — hold out *recording runs*
and *recording sessions* and hope those track the person signing. They are proxies
for signer-independence, not measurements of it.

That gap is now the single largest open question in the project. Five other levers
were tried and only two survived; the report's conclusion is that the model is
limited by data diversity that INCLUDE does not contain. **This recording session
is the experiment that tests it.**

Two numbers come out of it:

1. **How well the current model generalises to people it has never seen.** If
   ~35 % session-disjoint holds up on new signers, the protocol was sound. If it
   collapses to near-chance, then session-disjoint was *also* over-estimating, and
   the report's headline needs another correction.
2. **Whether new signers raise accuracy when trained on.** Only answerable after
   the first, and only meaningful against it.

Result (1) is worth having even if it is bad news. Especially if it is bad news.

## What to record

**50 words** — the INCLUDE-50 list, in [capture_wordlist.txt](capture_wordlist.txt).
That list is used rather than a fresh one so results are directly comparable with
the existing INCLUDE-50 benchmark numbers.

**Minimum viable session:**

| | count | why |
|---|---|---|
| signers | **3** | 2 gives no way to tell a signer-specific quirk from a general failure |
| words per signer | 50 | the full list |
| takes per word | 3 | ~450 clips total; enough for a ±2 pt standard error |
| time per signer | ~35 min | 150 clips at ~6 s plus resets |

Three signers × 35 minutes is one afternoon. That is the entire cost of answering
the project's biggest open question.

**If you can only get two signers, record them anyway** — the eval is weaker but
still informative. Do not wait for a perfect session.

## How to record

The goal is *deliberate variation*. INCLUDE's weakness is that everything in it
looks the same, so anything that makes your clips look different from each other
is a feature, not sloppiness.

**Vary between signers** — and ideally between takes:

- **Room and background.** Different walls, different clutter. Do not clear the
  background; a plain wall is what INCLUDE already has too much of.
- **Lighting.** Window light, ceiling light, lamp, evening. Uneven is fine.
- **Clothing.** Long and short sleeves especially — sleeves change hand contrast.
- **Distance and angle.** Roughly frontal, but shift a metre closer or farther and
  a few degrees off-axis between takes.

**Keep constant:**

- **Framing:** head and both hands in frame at all times, torso visible. Hands
  leaving frame is the one failure that makes a clip unusable.
- **Phone landscape**, braced or propped. 1080p if offered; 30 fps is fine.
- **Start and end at rest**, hands down, with roughly half a second of stillness at
  each end. The sign should occupy the middle of the clip.
- **One sign per clip**, 2–4 seconds.

Phone camera is genuinely fine. Detection runs at 720p internally, so a modern
phone exceeds what the pipeline uses.

## File layout

```
custom/
  priya/
    dog/
      take1.mp4
      take2.mp4
      take3.mp4
    loud/
      take1.mp4
      ...
  arjun/
    dog/
      ...
```

`custom/<signer>/<word>/<take>.mp4`. The **signer directory is the whole point** —
it is what INCLUDE lacks and what makes the split meaningful. Use a consistent
name per person; it never leaves the machine.

Word folder names must match [capture_wordlist.txt](capture_wordlist.txt) exactly.
Anything unrecognised is reported at ingest rather than silently dropped, since it
is usually a typo rather than a new sign.

## After recording

```bash
# 1. Extract crops (~40 min for 450 clips)
python -m islvit.data.crops --corpus custom --cache-dir cache_custom --crop-size 128

# 2. Build the signer-disjoint split -- new signers become test, INCLUDE is train
python -m islvit.data.ingest_custom --mode eval

# 3. Score the existing model on people it has never seen
python -m islvit.eval  --run runs/sd263_ssl_foldval \
                       --split-file splits/custom__signer-disjoint-eval.csv
python -m islvit.tta   --run runs/sd263_ssl_foldval \
                       --split-file splits/custom__signer-disjoint-eval.csv
```

Step 2 asserts that no signer appears on both sides of the split, so the protocol
cannot silently degrade the way the earlier ones did.

Then, to use the footage for training instead:

```bash
python -m islvit.data.ingest_custom --mode train --test-signers <one-name>
python -m islvit.train --config configs/full263_v2.yaml \
  --split-file splits/custom__signer-disjoint-train.csv --select last --tag custom_signer
```

## How to read the result

Against ~35 % on `session-disjoint`, scored on the same 50-word vocabulary:

| New-signer top-1 | Reading |
|---|---|
| **within ~5 pts** | session-disjoint was an honest proxy; the reported numbers stand |
| **10–20 pts lower** | expected; signer identity is a real channel the proxy missed |
| **near chance (2 %)** | the model keys on recording conditions, not handshape — everything strict in the report is still optimistic |

Run at least three seeds before believing any of it. The measured noise floor is
±1.9 points on top-1, and a single run on 450 clips will be noisier still.
