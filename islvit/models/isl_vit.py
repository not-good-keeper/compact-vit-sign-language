"""ISL-ViT-Tiny: a factorised video Vision Transformer over hand and face crops.

Structure follows ViViT's factorised-encoder variant, sized down hard:

    (B, T, S, 3, H, W)                  T timesteps x S streams of crops
        |
        |  STAGE A -- spatial ViT, weights shared across all T*S crops
        |             DeiT-Tiny geometry (dim 192, heads 3) so ImageNet
        |             weights load directly
        v
    (B, T*S, D)                         one embedding per crop
        |  + stream embedding, + time embedding, + missing-stream embedding
        |
        |  STAGE B -- temporal ViT over the T*S sequence with a CLS token
        v
    (B, n_classes)

Factorising matters at this size. Joint spatio-temporal attention over all
crops-patches would be O((T*S*P)^2); attending within a crop and then across
crops is O(P^2) + O((T*S)^2), which is what makes the model fit an edge budget.

The ImageNet initialisation of Stage A is the single biggest lever against
INCLUDE's ~16 clips per class, and is why the width is pinned to 192.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int, attn_drop: float = 0.0, proj_drop: float = 0.0) -> None:
        super().__init__()
        if dim % heads != 0:
            raise ValueError(f"dim {dim} must be divisible by heads {heads}")
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = attn_drop
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, tokens, dim = x.shape
        qkv = self.qkv(x).reshape(batch, tokens, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(0)
        dropout = self.attn_drop if self.training else 0.0
        out = torch.nn.functional.scaled_dot_product_attention(query, key, value, dropout_p=dropout)
        out = out.transpose(1, 2).reshape(batch, tokens, dim)
        return self.proj_drop(self.proj(out))


class DropPath(nn.Module):
    """Stochastic depth. Cheap regularisation and it matters a lot here."""

    def __init__(self, rate: float = 0.0) -> None:
        super().__init__()
        self.rate = rate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.rate == 0.0 or not self.training:
            return x
        keep = 1.0 - self.rate
        mask = x.new_empty((x.shape[0],) + (1,) * (x.ndim - 1)).bernoulli_(keep)
        return x * mask / keep


class Block(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: float = 4.0, drop_path: float = 0.0, drop: float = 0.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, heads, proj_drop=drop)
        self.drop_path = DropPath(drop_path)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(drop), nn.Linear(hidden, dim), nn.Dropout(drop))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x)))
        return x + self.drop_path(self.mlp(self.norm2(x)))


class SpatialEncoder(nn.Module):
    """ViT over a single crop, shared across every timestep and stream."""

    def __init__(
        self,
        img_size: int = 64,
        patch_size: int = 16,
        dim: int = 192,
        depth: int = 4,
        heads: int = 3,
        drop_path: float = 0.0,
        drop: float = 0.0,
    ) -> None:
        super().__init__()
        if img_size % patch_size != 0:
            raise ValueError(f"img_size {img_size} must be divisible by patch_size {patch_size}")
        self.grid = img_size // patch_size
        self.n_patches = self.grid**2
        self.dim = dim

        self.patch_embed = nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_patches + 1, dim))
        rates = torch.linspace(0, drop_path, depth).tolist()
        self.blocks = nn.ModuleList([Block(dim, heads, drop_path=rates[i], drop=drop) for i in range(depth)])
        self.norm = nn.LayerNorm(dim, eps=1e-6)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Patch-embed and add positions, without running the blocks.

        Split out so masked pretraining can substitute mask tokens between
        embedding and encoding. Returns (B, 1+P, D) with CLS at index 0.
        """
        batch = x.shape[0]
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        return torch.cat([self.cls_token.expand(batch, -1, -1), x], dim=1) + self.pos_embed

    def encode(self, tokens: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            tokens = block(tokens)
        return self.norm(tokens)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encode(self.embed(x))[:, 0]


class ISLViT(nn.Module):
    """Full model: spatial encoder over crops, temporal encoder over the sequence."""

    def __init__(
        self,
        n_classes: int,
        n_frames: int = 8,
        n_streams: int = 3,
        img_size: int = 64,
        patch_size: int = 16,
        dim: int = 192,
        spatial_depth: int = 4,
        temporal_depth: int = 4,
        heads: int = 3,
        drop_path: float = 0.1,
        drop: float = 0.0,
        landmarks: bool = False,
        lm_velocity: bool = False,
        lm_pair: bool = False,
        lm_wrist_vel: bool = False,
    ) -> None:
        super().__init__()
        self.n_frames = n_frames
        self.n_streams = n_streams
        self.landmarks = landmarks
        self.lm_velocity = lm_velocity
        self.lm_pair = lm_pair
        self.lm_wrist_vel = lm_wrist_vel

        self.spatial = SpatialEncoder(img_size, patch_size, dim, spatial_depth, heads, drop_path, drop)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.stream_embed = nn.Parameter(torch.zeros(1, n_streams, dim))
        self.time_embed = nn.Parameter(torch.zeros(1, n_frames, dim))
        # A stream that was never detected gets a black crop. Flagging it beats
        # letting the model infer "all-zero pixels" from the patch embedding --
        # the absence of a hand is itself informative.
        self.missing_embed = nn.Parameter(torch.zeros(1, 1, dim))
        # Cropping deletes where the hands are, but sign location is a defining
        # parameter of a word -- the same handshape at forehead and chest are
        # different signs. This projects (centre x, centre y, scale) back in.
        self.geometry_proj = nn.Sequential(nn.Linear(3, dim), nn.GELU(), nn.Linear(dim, dim))

        if landmarks:
            # Handshape measured directly rather than re-derived from 64 pixels:
            # each hand's 21 MediaPipe joints, made position- and scale-invariant,
            # plus where the wrist sits relative to the body. Pose goes to the face
            # token, which is the stream that already carries body context. Both
            # are added to the crop tokens like geometry, so the token count and
            # the temporal stage are unchanged.
            hand_in = HAND_FEATURES * (2 if lm_velocity else 1) + (2 if lm_wrist_vel else 0)
            pose_in = POSE_FEATURES + (3 if lm_pair else 0)
            self.hand_proj = nn.Sequential(nn.Linear(hand_in, dim), nn.GELU(), nn.Linear(dim, dim))
            self.pose_proj = nn.Sequential(nn.Linear(pose_in, dim), nn.GELU(), nn.Linear(dim, dim))

        rates = torch.linspace(0, drop_path, temporal_depth).tolist()
        self.blocks = nn.ModuleList([Block(dim, heads, drop_path=rates[i], drop=drop) for i in range(temporal_depth)])
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.head = nn.Linear(dim, n_classes)

        for parameter in (self.cls_token, self.stream_embed, self.time_embed, self.missing_embed):
            nn.init.trunc_normal_(parameter, std=0.02)
        # Start the geometry branch near zero so it refines the visual token
        # rather than swamping the pretrained features early in training.
        nn.init.zeros_(self.geometry_proj[-1].weight)
        nn.init.zeros_(self.geometry_proj[-1].bias)
        if landmarks:
            # Zero output, so at step 0 this *is* the pixel-only model and the
            # pretrained initialisation transfers untouched.
            for projection in (self.hand_proj, self.pose_proj):
                nn.init.zeros_(projection[-1].weight)
                nn.init.zeros_(projection[-1].bias)
        nn.init.zeros_(self.head.bias)
        nn.init.trunc_normal_(self.head.weight, std=0.01)

    def forward(
        self,
        crops: torch.Tensor,
        detected: torch.Tensor | None = None,
        geometry: torch.Tensor | None = None,
        **landmarks: torch.Tensor,
    ) -> torch.Tensor:
        """crops (B,T,S,3,H,W) float; detected (B,T,S) bool; geometry (B,T,S,3) float.

        ``landmarks`` is ``hands`` (B,T,2,21,3), ``hand_present`` (B,T,2) and
        ``pose`` (B,T,7,3), as produced by the dataset when landmarks are enabled.
        """
        return self.head(self.forward_features(crops, detected, geometry, **landmarks)[:, 0])

    def temporal(
        self,
        tokens: torch.Tensor,
        detected: torch.Tensor | None = None,
        geometry: torch.Tensor | None = None,
        hands: torch.Tensor | None = None,
        hand_present: torch.Tensor | None = None,
        pose: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Stage B over per-crop embeddings (B,T,S,D). Returns (B, 1+T*S, D).

        Taking pre-encoded crop embeddings rather than raw pixels lets masked
        pretraining reuse Stage B unchanged after it has already run Stage A on
        partially masked patches.
        """
        batch, frames, streams = tokens.shape[:3]
        tokens = tokens + self.stream_embed[:, None, :streams] + self.time_embed[:, :frames, None]

        if geometry is not None:
            tokens = tokens + self.geometry_proj(geometry)

        if detected is not None:
            tokens = tokens + (~detected).unsqueeze(-1).to(tokens.dtype) * self.missing_embed

        if self.landmarks:
            if hands is None:
                raise ValueError("this model was built with landmarks; pass hands, hand_present, pose")
            hand_features, pose_features, pose_ok, wrist = landmark_features(hands, pose)
            seen = hand_present.bool() & pose_ok.unsqueeze(-1)
            if self.lm_wrist_vel:
                # Where the wrist travels in the body frame since the previous
                # sampled frame. Only the wrist: differencing all 65 handshape
                # numbers mostly amplifies joint jitter, which is what made the
                # full-velocity variant worse.
                move = (wrist[:, 1:] - wrist[:, :-1]) * (seen[:, 1:] & seen[:, :-1]).unsqueeze(-1)
                move = torch.cat([torch.zeros_like(wrist[:, :1]), move], dim=1)
            if self.lm_pair:
                # The two hands' relation: right-minus-left wrist vector and its
                # length, in shoulder widths. Signs such as big / wide / narrow
                # differ almost only in this, and per-hand features never state it.
                both = (seen[..., 0] & seen[..., 1]).unsqueeze(-1).to(wrist.dtype)
                apart = wrist[:, :, 1] - wrist[:, :, 0]
                pair = torch.cat([apart, apart.norm(dim=-1, keepdim=True)], dim=-1) * both
                pose_features = torch.cat([pose_features, pair], dim=-1)
            if self.lm_velocity:
                # Explicit motion: each hand's feature change since the previous
                # sampled frame, zeroed unless the hand was seen in both frames.
                seen = hand_present.bool()
                step = hand_features[:, 1:] - hand_features[:, :-1]
                valid = (seen[:, 1:] & seen[:, :-1]).unsqueeze(-1).to(step.dtype)
                velocity = torch.cat([torch.zeros_like(hand_features[:, :1]), step * valid], dim=1)
                hand_features = torch.cat([hand_features, velocity], dim=-1)
            if self.lm_wrist_vel:
                hand_features = torch.cat([hand_features, move], dim=-1)
            hand_term = self.hand_proj(hand_features) * hand_present.unsqueeze(-1).to(hand_features.dtype)
            pose_term = self.pose_proj(pose_features) * pose_ok.unsqueeze(-1).to(pose_features.dtype)
            tokens = tokens + torch.cat([hand_term, pose_term.unsqueeze(2)], dim=2).to(tokens.dtype)

        tokens = tokens.reshape(batch, frames * streams, -1)
        tokens = torch.cat([self.cls_token.expand(batch, -1, -1), tokens], dim=1)
        for block in self.blocks:
            tokens = block(tokens)
        return self.norm(tokens)

    def forward_features(
        self,
        crops: torch.Tensor,
        detected: torch.Tensor | None = None,
        geometry: torch.Tensor | None = None,
        **landmarks: torch.Tensor,
    ) -> torch.Tensor:
        """Full encoder output (B, 1+T*S, D), CLS at index 0."""
        batch, frames, streams = crops.shape[:3]
        tokens = self.spatial(crops.flatten(0, 2)).reshape(batch, frames, streams, -1)
        return self.temporal(tokens, detected, geometry, **landmarks)


# Frames are letterboxed to 1280x720 before detection, so a normalised x unit is
# 16/9 of a normalised y unit. Undoing that makes the hand features isotropic --
# otherwise the same handshape rotated 90 degrees would read as a different shape.
ASPECT = 1280 / 720
HAND_FEATURES = 21 * 3 + 2
POSE_FEATURES = 7 * 3
# Pose order from islvit.data.landmarks: nose, L/R shoulder, L/R elbow, L/R wrist.
SHOULDERS = (1, 2)


def landmark_features(hands: torch.Tensor, pose: torch.Tensor):
    """Raw landmarks -> invariant features. Lives in the model, not the loader,
    so every inference path computes it identically.

    hands (B,T,2,21,3) -> (B,T,2,65): 21 joints relative to the wrist and divided
    by the hand's own extent (handshape, independent of where the hand is and how
    far it is from the camera), plus the wrist's position relative to the shoulder
    midpoint in shoulder widths (sign location, independent of framing).

    pose (B,T,7,3) -> (B,T,21): the same body-relative normalisation, visibility
    kept. ``pose_ok`` is False unless both shoulders were seen; without them there
    is no body frame, so every body-relative number is zeroed rather than divided
    by nothing.
    """
    hands, pose = hands.float(), pose.float()
    scale_xy = hands.new_tensor([ASPECT, 1.0])

    xyz = hands * hands.new_tensor([ASPECT, 1.0, 1.0])
    relative = xyz - xyz[..., :1, :]
    extent = relative[..., :2].norm(dim=-1).amax(dim=-1, keepdim=True).clamp(min=1e-3)
    shape = (relative / extent.unsqueeze(-1)).flatten(-2)

    pose_ok = (pose[..., SHOULDERS, 2] > 0.3).all(dim=-1)
    gate = pose_ok.unsqueeze(-1).float()
    left = pose[..., SHOULDERS[0], :2] * scale_xy
    right = pose[..., SHOULDERS[1], :2] * scale_xy
    centre = ((left + right) / 2).unsqueeze(-2)
    width = (left - right).norm(dim=-1, keepdim=True).clamp(min=1e-3).unsqueeze(-2)

    wrist = ((hands[..., 0, :2] * scale_xy - centre) / width) * gate.unsqueeze(-2)
    body = ((pose[..., :2] * scale_xy - centre) / width) * gate.unsqueeze(-2)
    pose_features = torch.cat([body, pose[..., 2:3]], dim=-1).flatten(-2)
    return torch.cat([shape, wrist], dim=-1), pose_features, pose_ok, wrist


def load_deit_tiny_weights(model: ISLViT, verbose: bool = True) -> ISLViT:
    """Initialise the spatial encoder from ImageNet DeiT/ViT-Tiny.

    Position embeddings are bicubically resized from the pretrained 14x14 grid to
    whatever grid our crop size implies, the standard ViT transfer procedure.
    Only the first ``spatial_depth`` blocks are taken.
    """
    import timm

    source = timm.create_model("vit_tiny_patch16_224.augreg_in21k_ft_in1k", pretrained=True, num_classes=0)
    source_state = source.state_dict()
    spatial = model.spatial
    loaded, skipped = [], []

    with torch.no_grad():
        spatial.patch_embed.weight.copy_(source_state["patch_embed.proj.weight"])
        spatial.patch_embed.bias.copy_(source_state["patch_embed.proj.bias"])
        spatial.cls_token.copy_(source_state["cls_token"])
        loaded += ["patch_embed", "cls_token"]

        source_pos = source_state["pos_embed"]
        cls_pos, grid_pos = source_pos[:, :1], source_pos[:, 1:]
        source_grid = int(math.sqrt(grid_pos.shape[1]))
        grid_pos = grid_pos.reshape(1, source_grid, source_grid, -1).permute(0, 3, 1, 2)
        grid_pos = torch.nn.functional.interpolate(
            grid_pos, size=(spatial.grid, spatial.grid), mode="bicubic", align_corners=False
        )
        grid_pos = grid_pos.permute(0, 2, 3, 1).reshape(1, spatial.grid**2, -1)
        spatial.pos_embed.copy_(torch.cat([cls_pos, grid_pos], dim=1))
        loaded.append(f"pos_embed ({source_grid}x{source_grid} -> {spatial.grid}x{spatial.grid})")

        for index, block in enumerate(spatial.blocks):
            prefix = f"blocks.{index}."
            mapping = {
                "norm1.weight": block.norm1.weight, "norm1.bias": block.norm1.bias,
                "attn.qkv.weight": block.attn.qkv.weight, "attn.qkv.bias": block.attn.qkv.bias,
                "attn.proj.weight": block.attn.proj.weight, "attn.proj.bias": block.attn.proj.bias,
                "norm2.weight": block.norm2.weight, "norm2.bias": block.norm2.bias,
                "mlp.fc1.weight": block.mlp[0].weight, "mlp.fc1.bias": block.mlp[0].bias,
                "mlp.fc2.weight": block.mlp[3].weight, "mlp.fc2.bias": block.mlp[3].bias,
            }
            for key, target in mapping.items():
                source_tensor = source_state.get(prefix + key)
                if source_tensor is not None and source_tensor.shape == target.shape:
                    target.copy_(source_tensor)
                else:
                    skipped.append(prefix + key)
            loaded.append(f"block {index}")

        spatial.norm.weight.copy_(source_state["norm.weight"])
        spatial.norm.bias.copy_(source_state["norm.bias"])
        loaded.append("norm")

    if verbose:
        print(f"Loaded DeiT-Tiny into spatial encoder: {len(loaded)} groups, {len(skipped)} skipped")
        if skipped:
            print(f"  skipped: {skipped[:8]}")
    return model


def count_parameters(model: nn.Module) -> dict[str, float]:
    total = sum(p.numel() for p in model.parameters())
    spatial = sum(p.numel() for p in model.spatial.parameters()) if hasattr(model, "spatial") else 0
    return {"total_M": total / 1e6, "spatial_M": spatial / 1e6, "temporal_M": (total - spatial) / 1e6}


if __name__ == "__main__":
    model = ISLViT(n_classes=50)
    stats = count_parameters(model)
    crops = torch.randn(2, 8, 3, 3, 64, 64)
    detected = torch.ones(2, 8, 3, dtype=torch.bool)
    geometry = torch.randn(2, 8, 3, 3) * 0.2
    logits = model(crops, detected, geometry)
    print(f"output {tuple(logits.shape)}")
    print(f"params total={stats['total_M']:.2f}M spatial={stats['spatial_M']:.2f}M temporal={stats['temporal_M']:.2f}M")
