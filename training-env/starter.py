"""The code every agent starts from: the model and one training step.
score.py trains it for exactly 500 steps and measures the loss on held-out text."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

VOCAB, CTX = 4096, 256  # fixed by the data
D, HEADS, LAYERS = 256, 4, 5
LR, WARMUP_FRAC, WD = 1e-3, 0.03, 0.1


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(D), nn.LayerNorm(D)
        self.qkv, self.proj = nn.Linear(D, 3 * D), nn.Linear(D, D)
        self.fc, self.out = nn.Linear(D, 4 * D), nn.Linear(4 * D, D)

    def forward(self, x):
        B, T, _ = x.shape
        q, k, v = self.qkv(self.n1(x)).view(B, T, 3, HEADS, D // HEADS).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(a.transpose(1, 2).reshape(B, T, D))
        return x + self.out(F.gelu(self.fc(self.n2(x))))


class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok, self.pos = nn.Embedding(VOCAB, D), nn.Embedding(CTX, D)
        self.blocks = nn.ModuleList(Block() for _ in range(LAYERS))
        self.norm = nn.LayerNorm(D)
        self.head = nn.Linear(D, VOCAB, bias=False)
        self.head.weight = self.tok.weight  # tied
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)
            if getattr(m, "bias", None) is not None:
                nn.init.zeros_(m.bias)

    def forward(self, idx):
        """idx: (batch, time) token ids. Returns logits (batch, time, VOCAB)."""
        x = self.tok(idx) + self.pos(torch.arange(idx.size(1), device=idx.device))
        for b in self.blocks:
            x = b(x)
        return self.head(self.norm(x))


def build_model():
    return GPT()


def build_optimizer(model):
    return torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD, betas=(0.9, 0.95))


def train_step(model, opt, x, y, frac):
    """One update on one batch. x, y: (32, 256) token ids, y is x shifted by one.
    frac: share of training already done, from 0 to 1. Returns the loss."""
    lr = LR * frac / WARMUP_FRAC if frac < WARMUP_FRAC else \
        LR * 0.5 * (1 + math.cos(math.pi * (frac - WARMUP_FRAC) / (1 - WARMUP_FRAC)))
    for g in opt.param_groups:
        g["lr"] = lr
    loss = F.cross_entropy(model(x).view(-1, VOCAB), y.reshape(-1))
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    return loss
