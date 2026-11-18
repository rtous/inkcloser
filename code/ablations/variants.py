"""Ablation variants of InkCloser (EXP-1 of the response to reviewers).

Each variant = a network (the full LineInpainter with components removed or
replaced) + loss settings (passed to train.py as command-line flags).
"""
import os
import sys

import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from inkcloser.model import GatedConv2d, LineInpainter  # noqa: E402

PAPER_LOSS = []                                                       # λh=3, β=(1, 1, 0.5)
BCE_ONLY = ["--hole-weight", "1", "--dice-weight", "0", "--gap-dice-weight", "0"]
NO_GAP_DICE = ["--gap-dice-weight", "0"]

# key: (label, model options, extra train.py args)
VARIANTS = {
    "full":                 ("InkCloser (full)",          dict(), PAPER_LOSS),
    "no_attn28":            ("w/o 28×28 attention",       dict(attn28=False), PAPER_LOSS),
    "no_branch56":          ("w/o 56×56 branch",          dict(branch56=False), PAPER_LOSS),
    "no_gated":             ("w/o gated conv",            dict(gated=False), PAPER_LOSS),
    "no_gap_dice":          ("w/o local (gap) Dice",      dict(), NO_GAP_DICE),
    "plain_unet_ours_loss": ("Plain U-Net + our loss",    dict(gated=False, attn28=False, branch56=False), PAPER_LOSS),
    "plain_unet_bce":       ("Plain U-Net + BCE",         dict(gated=False, attn28=False, branch56=False), BCE_ONLY),
}


def _ungate(module):
    """Replace every GatedConv2d by a standard Conv2d of the same shape."""
    for name, child in module.named_children():
        if isinstance(child, GatedConv2d):
            f = child.feature
            setattr(module, name, nn.Conv2d(f.in_channels, f.out_channels, f.kernel_size,
                                            f.stride, f.padding))
        else:
            _ungate(child)


def build_model(gated=True, attn28=True, branch56=True):
    """LineInpainter with components removed. Removed modules become nn.Identity,
    so the U-Net (encoder, skips, decoder, head) is unchanged otherwise."""
    m = LineInpainter()
    if not gated:
        _ungate(m.encoder)
    if not attn28:
        m.bottleneck_attn = nn.Identity()
    if not branch56:
        m.mid_dilated = nn.Identity()
        m.mid_attn = nn.Identity()
    return m


def model_fn(key):
    opts = VARIANTS[key][1]
    return lambda: build_model(**opts)
