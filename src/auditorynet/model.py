"""AuditoryNet architecture (Section 3.4)."""
from __future__ import annotations

import torch
import torch.nn as nn


def _split_channels(total: int, parts: int = 3) -> list:
    base, rem = divmod(total, parts)
    return [base + (1 if i < rem else 0) for i in range(parts)]


class A1Block(nn.Module):
    """Primary auditory cortex module: Y = [P3 || P5 || P7](X) + Conv1x1(X).

    The three parallel branches share ``out_ch`` between them (32 -> 11/11/10) so
    the concatenation has exactly ``out_ch`` channels and can be summed with the
    1x1 skip projection (Eqs. 3.7-3.9).
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        chans = _split_channels(out_ch)
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(in_ch, c, k, padding=k // 2, bias=False),
                    nn.BatchNorm2d(c),
                    nn.GELU(),
                )
                for k, c in zip((3, 5, 7), chans)
            ]
        )
        self.skip = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([b(x) for b in self.branches], dim=1) + self.skip(x)


class BeltBlock(nn.Module):
    """Auditory belt module: depthwise-separable residual unit that integrates
    5 steps along TIME and 3 along FREQUENCY (Eq. 3.10).

    Tensors are (B, C, freq, time), so the kernel is (3, 5) in (H, W).
    """

    def __init__(self, channels: int):
        super().__init__()
        self.dw = nn.Conv2d(channels, channels, (3, 5), padding=(1, 2), groups=channels, bias=False)
        self.pw = nn.Conv2d(channels, channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(channels)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.pw(self.dw(x)))) + x


class AuditoryNet(nn.Module):
    """Input (B, 1, 64, 128) cochleagram -> logits (B, 2) ordered [HC, PD]."""

    def __init__(self, dropout: float = 0.4, n_classes: int = 2):
        super().__init__()
        self.a1_1 = A1Block(1, 32)
        self.pool1 = nn.MaxPool2d(2)
        self.gn1 = nn.GroupNorm(8, 32)
        self.a1_2 = A1Block(32, 64)
        self.pool2 = nn.MaxPool2d(2)
        self.gn2 = nn.GroupNorm(8, 64)
        self.belt = nn.Sequential(BeltBlock(64), BeltBlock(64))
        self.gap = nn.AdaptiveAvgPool2d((4, 8))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 8, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(64, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.gn1(self.pool1(self.a1_1(x)))
        x = self.gn2(self.pool2(self.a1_2(x)))
        x = self.belt(x)
        return self.classifier(self.gap(x))


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
