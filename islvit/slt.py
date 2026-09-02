"""Sign-language translation scaffolding: ISL video -> English text.

**Preliminary by design.** iSign's own paper reports BLEU-4 = 0.56 for
SignVideo2Text -- essentially non-functional -- so the goal here is a correct,
runnable pipeline to build on, not a working translator. Treat any BLEU this
produces as a smoke test that gradients flow and the decoder learns *something*,
not as a quality claim.

Architecture reuses the isolated-word encoder unchanged::

    crops (B,T,S,3,H,W)
        -> ISLViT.forward_features  -> (B, 1+T*S, D)   memory
        -> TransformerDecoder cross-attending to that memory
        -> English word tokens

Using ``forward_features`` rather than the pooled CLS vector is the whole point:
translation needs to attend to *when* something was signed, and pooling to a
single vector throws that away.

**Known limitation, stated up front:** the crop cache stores 16 frames per clip,
which was sized for one isolated sign. iSign captions run a median of 9 words, so
this is ~1.8 frames per signed word -- far too coarse to resolve handshape
transitions. Fixing it means re-extracting iSign at 32-64 frames (~1.5 h
re-download plus disk we do not currently have). Architecture first, resolution
second.

Usage::

    python -m islvit.slt --epochs 20 --limit 4000      # quick scaffold check
    python -m islvit.slt --epochs 60                   # full 18k
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from islvit.data.dataset import SOURCE_DIRECT, SOURCE_ROI, prepare_clip
from islvit.models.isl_vit import ISLViT

CACHE = Path("cache_isign")
CAPTIONS = Path("iSign_meta/iSign_v1.1.csv")
RUNS_DIR = Path("runs")
PAD, BOS, EOS, UNK = 0, 1, 2, 3

WORD_RE = re.compile(r"[a-z0-9']+")


def tokenize(text: str) -> list[str]:
    return WORD_RE.findall(text.lower())


class Vocab:
    """Word-level vocabulary.

    18k captions cannot support 15k word types, so rare words collapse to
    ``<unk>`` rather than becoming untrainable singleton classes.
    """

    def __init__(self, texts: list[str], min_freq: int = 3, max_size: int = 4000) -> None:
        counts = Counter(word for text in texts for word in tokenize(text))
        kept = [w for w, c in counts.most_common() if c >= min_freq][: max_size - 4]
        self.itos = ["<pad>", "<bos>", "<eos>", "<unk>"] + kept
        self.stoi = {w: i for i, w in enumerate(self.itos)}
        covered = sum(counts[w] for w in kept)
        total = max(1, sum(counts.values()))
        print(f"  vocab {len(self.itos)} types (min_freq={min_freq}), token coverage {covered/total:.1%}")

    def __len__(self) -> int:
        return len(self.itos)

    def encode(self, text: str, max_len: int) -> list[int]:
        ids = [self.stoi.get(w, UNK) for w in tokenize(text)][: max_len - 2]
        return [BOS] + ids + [EOS]

    def decode(self, ids) -> str:
        out = []
        for i in ids:
            i = int(i)
            if i in (EOS, PAD):
                break
            if i != BOS:
                out.append(self.itos[i] if i < len(self.itos) else "<unk>")
        return " ".join(out)


class SignTranslation(Dataset):
    def __init__(self, rows, vocab, n_frames=8, img_size=64, max_len=24, train=True):
        self.rows, self.vocab, self.n_frames = rows, vocab, n_frames
        self.img_size, self.max_len, self.train = img_size, max_len, train
        self._memmaps = None

    def __len__(self):
        return len(self.rows)

    def _cache(self):
        if self._memmaps is None:
            self._memmaps = tuple(
                np.load(CACHE / name, mmap_mode="r")
                for name in ("crops.npy", "sources.npy", "geometry.npy")
            )
        return self._memmaps

    def __getitem__(self, index):
        crops_cache, sources_cache, geometry_cache = self._cache()
        row, text = self.rows[index]
        clip = np.asarray(crops_cache[row])
        sources = np.asarray(sources_cache[row])
        geometry = np.asarray(geometry_cache[row]).astype(np.float32)
        detected = (sources == SOURCE_DIRECT) | (sources == SOURCE_ROI)

        edges = np.linspace(0, clip.shape[0], self.n_frames + 1)
        offsets = np.random.rand(self.n_frames) if self.train else np.full(self.n_frames, 0.5)
        picks = np.clip((edges[:-1] + offsets * (edges[1:] - edges[:-1])).astype(int), 0, clip.shape[0] - 1)

        crops, detected, geometry = prepare_clip(
            clip[picks],
            detected[picks],
            geometry[picks],
            img_size=self.img_size,
            train=self.train,
            color_jitter=0.3 if self.train else 0.0,
            # No horizontal flip: it mirrors the signer, but the caption is not
            # mirrored with it. Harmless for a symmetric class label, wrong for
            # a directional one (pointing, role shift, spatial reference).
            flip_prob=0.0,
            stream_dropout=0.0,
            resolution_jitter=0.5 if self.train else 0.0,
        )
        ids = self.vocab.encode(text, self.max_len)
        ids = ids + [PAD] * (self.max_len - len(ids))
        return {
            "crops": torch.from_numpy(crops),
            "detected": torch.from_numpy(detected),
            "geometry": torch.from_numpy(geometry),
            "tokens": torch.tensor(ids, dtype=torch.long),
        }


class Translator(nn.Module):
    """ISLViT encoder + a small transformer decoder."""

    def __init__(self, vocab_size, n_frames=8, img_size=64, dim=192, depth=4, heads=3, max_len=24):
        super().__init__()
        # n_classes is unused -- the classification head is discarded and
        # forward_features is the interface -- but ISLViT requires the argument.
        self.encoder = ISLViT(n_classes=1, n_frames=n_frames, img_size=img_size, dim=dim)
        self.embed = nn.Embedding(vocab_size, dim, padding_idx=PAD)
        self.pos = nn.Parameter(torch.zeros(1, max_len, dim))
        nn.init.trunc_normal_(self.pos, std=0.02)
        layer = nn.TransformerDecoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=dim * 4,
            dropout=0.1,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=depth)
        self.out = nn.Linear(dim, vocab_size)

    def forward(self, crops, detected, geometry, tokens):
        memory = self.encoder.forward_features(crops, detected, geometry)
        x = self.embed(tokens) + self.pos[:, : tokens.size(1)]
        # Causal mask: position i must not attend to future tokens, or the model
        # trivially copies the answer it is being asked to predict.
        causal = nn.Transformer.generate_square_subsequent_mask(tokens.size(1), device=tokens.device)
        x = self.decoder(x, memory, tgt_mask=causal, tgt_key_padding_mask=(tokens == PAD))
        return self.out(x)

    @torch.no_grad()
    def generate(self, crops, detected, geometry, max_len=24):
        memory = self.encoder.forward_features(crops, detected, geometry)
        ids = torch.full((crops.size(0), 1), BOS, dtype=torch.long, device=crops.device)
        finished = torch.zeros(crops.size(0), dtype=torch.bool, device=crops.device)
        for _ in range(max_len - 1):
            x = self.embed(ids) + self.pos[:, : ids.size(1)]
            causal = nn.Transformer.generate_square_subsequent_mask(ids.size(1), device=ids.device)
            nxt = self.out(self.decoder(x, memory, tgt_mask=causal))[:, -1].argmax(-1, keepdim=True)
            ids = torch.cat([ids, nxt], dim=1)
            finished |= nxt.squeeze(1) == EOS
            if finished.all():
                break
        return ids


def bleu(hyps: list[str], refs: list[str], max_n: int = 4) -> float:
    """Corpus BLEU-4 with brevity penalty (sacrebleu is not installed).

    ``refs`` must be the ORIGINAL caption text, never the vocabulary round-trip.
    Decoding the reference through the vocabulary turns its rare words into
    ``<unk>``, and a model emitting ``<unk>`` then scores perfect n-gram matches
    against them: the first smoke test reported BLEU-4 8.04 -- above the
    published baseline -- purely from ``<unk>`` matching ``<unk>``.
    """
    precisions, hyp_len, ref_len = [], 0, 0
    for n in range(1, max_n + 1):
        match = total = 0
        for hyp, ref in zip(hyps, refs):
            h, r = hyp.split(), ref.split()
            if n == 1:
                hyp_len += len(h)
                ref_len += len(r)
            hg = Counter(tuple(h[i : i + n]) for i in range(len(h) - n + 1))
            rg = Counter(tuple(r[i : i + n]) for i in range(len(r) - n + 1))
            match += sum((hg & rg).values())
            total += max(0, len(h) - n + 1)
        precisions.append(match / total if total else 0.0)
    if min(precisions) == 0:
        return 0.0
    geometric = math.exp(sum(math.log(p) for p in precisions) / max_n)
    penalty = 1.0 if hyp_len > ref_len else math.exp(1 - ref_len / max(1, hyp_len))
    return 100 * penalty * geometric


def main() -> None:
    parser = argparse.ArgumentParser(description="ISL -> English translation scaffold")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--limit", type=int, default=0, help="use only the first N clips (smoke test)")
    parser.add_argument("--init-from", type=str, default=None, help="encoder weights, e.g. runs/isign_mim/pretrained.pt")
    parser.add_argument("--tag", type=str, default="slt")
    parser.add_argument("--max-len", type=int, default=24)
    args = parser.parse_args()

    caption = {row["uid"]: row["text"] for row in csv.DictReader(CAPTIONS.open(encoding="utf-8"))}
    with (CACHE / "index.csv").open(encoding="utf-8") as handle:
        rows = [
            (int(r["row"]), caption[r["video_path"][:-4]])
            for r in csv.DictReader(handle)
            if r["cached"] == "1" and r["video_path"][:-4] in caption
        ]
    if args.limit:
        rows = rows[: args.limit]
    print(f"[{args.tag}] {len(rows)} (clip, caption) pairs")

    rng = np.random.default_rng(0)
    order = rng.permutation(len(rows))
    cut = int(len(rows) * 0.9)
    train_rows = [rows[i] for i in order[:cut]]
    test_rows = [rows[i] for i in order[cut:]]

    vocab = Vocab([text for _, text in train_rows])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_set = SignTranslation(train_rows, vocab, max_len=args.max_len, train=True)
    test_set = SignTranslation(test_rows, vocab, max_len=args.max_len, train=False)
    print(f"  train {len(train_set)}  test {len(test_set)}  device {device}")

    model = Translator(len(vocab), max_len=args.max_len).to(device)
    if args.init_from:
        checkpoint = torch.load(args.init_from, map_location="cpu", weights_only=False)
        state = {k: v for k, v in checkpoint["backbone"].items() if not k.startswith("head.")}
        missing, unexpected = model.encoder.load_state_dict(state, strict=False)
        if unexpected:
            raise SystemExit(f"unexpected keys in {args.init_from}: {sorted(unexpected)[:5]}")
        print(f"  encoder initialised from {args.init_from} ({len(state)} tensors, {len(missing)} missing)")

    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    encoder_params = sum(p.numel() for p in model.encoder.parameters()) / 1e6
    print(f"  params total {total_params:.2f}M (encoder {encoder_params:.2f}M, decoder {total_params-encoder_params:.2f}M)")

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, num_workers=2)
    # Raw reference text, in the same order the unshuffled test loader yields.
    test_refs = [" ".join(tokenize(text)) for _, text in test_rows]
    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    steps = args.epochs * max(1, len(train_loader))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimiser, max_lr=args.lr, total_steps=steps, pct_start=0.1)
    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD, label_smoothing=0.1)

    output_dir = RUNS_DIR / args.tag
    output_dir.mkdir(parents=True, exist_ok=True)
    history = (output_dir / "history.csv").open("w", newline="", encoding="utf-8")
    writer = csv.writer(history)
    writer.writerow(["epoch", "train_loss", "test_loss", "bleu4", "seconds"])

    score = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        started, running, seen = time.time(), 0.0, 0
        for batch in train_loader:
            tokens = batch["tokens"].to(device)
            logits = model(
                batch["crops"].to(device),
                batch["detected"].to(device),
                batch["geometry"].to(device),
                tokens[:, :-1],
            )
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), tokens[:, 1:].reshape(-1))
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            scheduler.step()
            running += loss.item() * tokens.size(0)
            seen += tokens.size(0)

        model.eval()
        hyps, test_loss, test_n = [], 0.0, 0
        with torch.no_grad():
            for batch in test_loader:
                tokens = batch["tokens"].to(device)
                crops, detected, geometry = (batch[k].to(device) for k in ("crops", "detected", "geometry"))
                logits = model(crops, detected, geometry, tokens[:, :-1])
                test_loss += loss_fn(
                    logits.reshape(-1, logits.size(-1)), tokens[:, 1:].reshape(-1)
                ).item() * tokens.size(0)
                test_n += tokens.size(0)
                generated = model.generate(crops, detected, geometry, max_len=args.max_len)
                hyps += [vocab.decode(g.tolist()) for g in generated]
        # References are the raw captions, taken in loader order (test_loader is
        # unshuffled), never the vocabulary round-trip -- see bleu()'s docstring.
        score = bleu(hyps, test_refs)
        writer.writerow(
            [epoch, f"{running/max(1,seen):.4f}", f"{test_loss/max(1,test_n):.4f}",
             f"{score:.2f}", f"{time.time()-started:.1f}"]
        )
        history.flush()
        if epoch % 5 == 0 or epoch == 1:
            print(
                f"  ep {epoch:3d}  train {running/max(1,seen):.3f}  "
                f"test {test_loss/max(1,test_n):.3f}  BLEU-4 {score:.2f}  {time.time()-started:.0f}s",
                flush=True,
            )
            if epoch % 20 == 0 or epoch == 1:
                for hyp, ref in list(zip(hyps, test_refs))[:2]:
                    print(f"     ref: {ref[:80]}\n     hyp: {hyp[:80]}", flush=True)

    history.close()
    torch.save({"model": model.state_dict(), "vocab": vocab.itos, "args": vars(args)}, output_dir / "slt.pt")
    (output_dir / "summary.json").write_text(
        json.dumps(
            {
                "tag": args.tag,
                "task": "slt",
                "pairs": len(rows),
                "vocab": len(vocab),
                "params_M": round(total_params, 3),
                "bleu4": round(score, 2),
                "note": "preliminary; iSign paper baseline BLEU-4 is 0.56",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n[{args.tag}] final BLEU-4 {score:.2f}  (iSign paper baseline: 0.56)")


if __name__ == "__main__":
    main()
