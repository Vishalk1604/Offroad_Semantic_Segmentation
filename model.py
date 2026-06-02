"""
Model: frozen DINOv2 (vision foundation backbone) + Rein-style adapters + DPT decoder.

  - Backbone: HuggingFace `Dinov2Model`, fully frozen and run under no_grad. Its
    self-supervised features generalize across domains, which is what the unseen-test-
    environment score depends on.
  - Rein adapters: lightweight learnable-token refinement applied to the tapped patch
    tokens (parameter-efficient domain adaptation; only a few M trainable params). This
    is a memory-friendly take on Rein (CVPR'24) that keeps the backbone fully frozen so
    it fits 8 GB. Swap to LoRA via `peft` if desired — same story.
  - Decoder: DPT-style multi-scale reassemble + RefineNet fusion head.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Dinov2Model

from config import BACKBONES, PATCH_SIZE


# ---------------------------------------------------------------------------
# Rein-style adapter
# ---------------------------------------------------------------------------
class ReinAdapter(nn.Module):
    """Refine frozen patch tokens via cross-attention to a small set of learnable
    tokens. Output projection is zero-initialized so the adapter starts as identity."""

    def __init__(self, dim: int, num_tokens: int = 100):
        super().__init__()
        self.tokens = nn.Parameter(torch.randn(num_tokens, dim) * 0.02)
        self.norm = nn.LayerNorm(dim)
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)
        self.scale = dim ** -0.5
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x):                       # x: (B, N, C)
        h = self.norm(x)
        q = self.q(h)                           # (B, N, C)
        k = self.k(self.tokens)                 # (T, C)
        v = self.v(self.tokens)                 # (T, C)
        attn = torch.softmax((q @ k.t()) * self.scale, dim=-1)  # (B, N, T)
        delta = attn @ v                        # (B, N, C)
        return x + self.proj(delta)


# ---------------------------------------------------------------------------
# DPT decoder
# ---------------------------------------------------------------------------
class ResidualConvUnit(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.act = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(c, c, 3, padding=1)
        self.conv2 = nn.Conv2d(c, c, 3, padding=1)

    def forward(self, x):
        return x + self.conv2(self.act(self.conv1(self.act(x))))


class FeatureFusionBlock(nn.Module):
    """Fuse a coarse path with a finer skip feature, resizing to the skip's size
    (handles odd ViT grid sizes cleanly)."""

    def __init__(self, c):
        super().__init__()
        self.rcu_skip = ResidualConvUnit(c)
        self.rcu_out = ResidualConvUnit(c)
        self.out_conv = nn.Conv2d(c, c, 1)

    def forward(self, x, skip=None):
        if skip is not None:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = x + self.rcu_skip(skip)
        x = self.rcu_out(x)
        return self.out_conv(x)


class DPTHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int, features: int = 256):
        super().__init__()
        self.proj = nn.ModuleList([nn.Conv2d(in_dim, features, 1) for _ in range(4)])
        # multi-scale reassemble: 4x, 2x, 1x, 0.5x relative to the patch grid
        self.resample = nn.ModuleList([
            nn.ConvTranspose2d(features, features, 4, stride=4),
            nn.ConvTranspose2d(features, features, 2, stride=2),
            nn.Identity(),
            nn.Conv2d(features, features, 3, stride=2, padding=1),
        ])
        self.fusion = nn.ModuleList([FeatureFusionBlock(features) for _ in range(4)])
        self.head = nn.Sequential(
            nn.Conv2d(features, features, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, num_classes, 1),
        )

    def forward(self, feats, out_size):
        f = [self.resample[i](self.proj[i](feats[i])) for i in range(4)]
        x = self.fusion[3](f[3])           # smallest scale
        x = self.fusion[2](x, f[2])
        x = self.fusion[1](x, f[1])
        x = self.fusion[0](x, f[0])        # largest scale
        x = self.head(x)
        return F.interpolate(x, size=out_size, mode="bilinear", align_corners=False)


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------
class OffroadSegModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        model_id, dim, depth = BACKBONES[cfg.backbone]
        self.backbone = Dinov2Model.from_pretrained(model_id)
        self.backbone.requires_grad_(False)
        self.backbone.eval()

        self.taps = [depth // 4, depth // 2, 3 * depth // 4, depth]  # hidden_states indices
        self.gh = cfg.img_h // PATCH_SIZE
        self.gw = cfg.img_w // PATCH_SIZE
        self.use_rein = cfg.use_rein
        if self.use_rein:
            self.adapters = nn.ModuleList([ReinAdapter(dim, cfg.rein_tokens) for _ in self.taps])
        self.head = DPTHead(dim, cfg.num_classes)

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()   # keep frozen backbone in eval mode always
        return self

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def forward(self, x):
        out_size = x.shape[-2:]
        # token grid is derived from the actual input size, not a fixed train-time size,
        # so the model works at any resolution (e.g. TTA multi-scale or a different --img-h/w).
        gh, gw = x.shape[-2] // PATCH_SIZE, x.shape[-1] // PATCH_SIZE
        with torch.no_grad():
            hidden = self.backbone(x, output_hidden_states=True).hidden_states
        feats = []
        for i, t in enumerate(self.taps):
            tok = hidden[t][:, 1:, :]                  # drop CLS token -> (B, N, C)
            tok = tok.float()                          # decoder runs in fp32
            if self.use_rein:
                tok = self.adapters[i](tok)
            b, n, c = tok.shape
            fmap = tok.transpose(1, 2).reshape(b, c, gh, gw)
            feats.append(fmap)
        return self.head(feats, out_size)


def build_model(cfg):
    model = OffroadSegModel(cfg)
    n_train = sum(p.numel() for p in model.trainable_parameters())
    n_total = sum(p.numel() for p in model.parameters())
    print(f"Model: DINOv2-{cfg.backbone} (frozen) + "
          f"{'Rein adapters + ' if cfg.use_rein else ''}DPT head")
    print(f"  Trainable params: {n_train/1e6:.2f}M / {n_total/1e6:.2f}M total")
    return model
