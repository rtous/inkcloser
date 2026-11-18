"""InkCloser network (CODE BLOCKS 6-9 of the notebook).

Module and attribute names are unchanged, so checkpoints saved by the notebook
(and the released weights) load with load_state_dict().
"""
import numpy as np
import torch
import torch.nn as nn


def gn(num_channels, num_groups=8):
    """GroupNorm with safe group count (must divide num_channels)."""
    while num_channels % num_groups != 0:
        num_groups -= 1
    return nn.GroupNorm(num_groups, num_channels)


# CODE BLOCK 6 (sinusoidal embedding)
class SinusoidalPosEmbed2D(nn.Module):
    def __init__(self, dim, h, w):
        super().__init__()
        assert dim % 4 == 0, "dim must be divisible by 4"
        pe_h = self._make_pe(h, dim // 2)   # (H, dim/2)
        pe_w = self._make_pe(w, dim // 2)   # (W, dim/2)
        pe = torch.cat([pe_h.unsqueeze(1).expand(-1, w, -1),
                        pe_w.unsqueeze(0).expand(h, -1, -1)], dim=-1)  # (H, W, dim)
        self.register_buffer('pe', pe.reshape(1, h * w, dim))

    def _make_pe(self, length, dim):
        pos = torch.arange(length).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, dim, 2).float() * -(np.log(10000.0) / dim))
        pe = torch.zeros(length, dim)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return pe

    def forward(self, x):
        return x + self.pe


# CODE BLOCK 7 (attention)
class TransformerBlock(nn.Module):
    def __init__(self, dim, heads=8, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
        )

    def forward(self, x):
        normed = self.norm1(x)
        x = x + self.attn(normed, normed, normed, need_weights=False)[0]   # no full attention matrix
        x = x + self.mlp(self.norm2(x))
        return x


class BottleneckAttention(nn.Module):
    def __init__(self, dim=512, heads=8, num_blocks=6, spatial_size=28):
        super().__init__()
        self.pos_embed = SinusoidalPosEmbed2D(dim, spatial_size, spatial_size)
        self.blocks = nn.Sequential(
            *[TransformerBlock(dim, heads) for _ in range(num_blocks)]
        )
        self.norm_out = nn.LayerNorm(dim)                     # rescales output before the decoder

    def forward(self, x):
        B, C, H, W = x.shape
        with torch.autocast('cuda', enabled=False):          # everything in here runs in float32
            tokens = x.float().flatten(2).permute(0, 2, 1)
            tokens = self.pos_embed(tokens)
            tokens = self.blocks(tokens)
            tokens = self.norm_out(tokens)                    # apply the exit normalization
            return tokens.permute(0, 2, 1).reshape(B, C, H, W)


# CODE BLOCK 8 (dilated residual stack)
class DilatedResBlock(nn.Module):
    """Single residual block with one dilated 3x3 conv."""

    def __init__(self, channels, dilation):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3,
                               padding=dilation, dilation=dilation)
        self.norm1 = gn(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, dilation=1)
        self.norm2 = gn(channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.act(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        return self.act(out + residual)


class DilatedContextStack(nn.Module):
    """Residual blocks with increasing dilation, applied at the 56x56 feature map."""

    def __init__(self, channels, dilations=(1, 2, 4, 8, 16)):
        super().__init__()
        self.blocks = nn.Sequential(*[DilatedResBlock(channels, d) for d in dilations])

    def forward(self, x):
        return self.blocks(x)


# CODE BLOCK 9 (U-Net)
class GatedConv2d(nn.Module):
    """Gated convolution: learns a soft, per-pixel mask of where the input is reliable."""

    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0):
        super().__init__()
        self.feature = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding)
        self.gate = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding)

    def forward(self, x):
        feat = self.feature(x)
        mask = torch.sigmoid(self.gate(x))
        return feat * mask


def _enc_stage(in_ch, out_ch, down):
    k, s = (4, 2) if down else (3, 1)
    return nn.Sequential(
        GatedConv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=1), gn(out_ch), nn.ReLU(),
        GatedConv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1), gn(out_ch), nn.ReLU(),
    )


class UNetEncoder(nn.Module):
    def __init__(self, c0=16):
        super().__init__()
        self.enc0 = _enc_stage(1, c0, down=False)     # 896, no downsampling
        self.enc1 = _enc_stage(c0, 32, down=True)     # 448
        self.enc2 = _enc_stage(32, 64, down=True)     # 224
        self.enc3 = _enc_stage(64, 128, down=True)    # 112
        self.enc4 = _enc_stage(128, 256, down=True)   # 56
        self.enc5 = _enc_stage(256, 512, down=True)   # 28

    def forward(self, x):
        e0 = self.enc0(x)
        e1 = self.enc1(e0)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)
        return e5, [e0, e1, e2, e3, e4]


def _dec_stage(in_ch, out_ch):
    return nn.Sequential(
        nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1), gn(out_ch), nn.ReLU(),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1), gn(out_ch), nn.ReLU(),
    )


class UNetDecoder(nn.Module):
    def __init__(self, c0=16):
        super().__init__()
        self.dec1 = _dec_stage(512, 256)
        self.dec2 = _dec_stage(512, 128)
        self.dec3 = _dec_stage(256, 64)
        self.dec4 = _dec_stage(128, 32)
        # Upsample only: 448 -> 896
        self.dec5 = nn.Sequential(
            nn.ConvTranspose2d(64, 16, kernel_size=4, stride=2, padding=1), gn(16), nn.ReLU(),
        )
        # Full-res head after concatenating the e0 skip
        self.head = nn.Sequential(
            nn.Conv2d(16 + c0, 16, kernel_size=3, padding=1), gn(16), nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1), gn(16), nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=3, padding=1),   # logits
        )

    def forward(self, x, skip_connections):
        e0, e1, e2, e3, e4 = skip_connections
        d1 = self.dec1(x)                                  # 56
        d2 = self.dec2(torch.cat([d1, e4], dim=1))         # 112
        d3 = self.dec3(torch.cat([d2, e3], dim=1))         # 224
        d4 = self.dec4(torch.cat([d3, e2], dim=1))         # 448
        d5 = self.dec5(torch.cat([d4, e1], dim=1))         # 896
        return self.head(torch.cat([d5, e0], dim=1))


class LineInpainter(nn.Module):
    """Returns logits in train mode and sigmoid probabilities (1 = paper) in eval mode."""

    def __init__(self):
        super().__init__()
        self.encoder = UNetEncoder()
        # global reasoning at the coarsest scale (28x28)
        self.bottleneck_attn = BottleneckAttention(dim=512, heads=8, num_blocks=6, spatial_size=28)
        # dilated context stack at 56x56, before mid attention
        self.mid_dilated = DilatedContextStack(channels=256, dilations=(1, 2, 4, 8, 16))
        # mid attention at 56x56 (each token = 16x16 patch)
        self.mid_attn = BottleneckAttention(dim=256, heads=8, num_blocks=3, spatial_size=56)
        self.decoder = UNetDecoder()

    def forward(self, x):
        bottleneck, skips = self.encoder(x)
        e0, e1, e2, e3, e4 = skips
        bottleneck = self.bottleneck_attn(bottleneck)
        e4 = self.mid_dilated(e4)      # 56x56
        e4 = self.mid_attn(e4)
        output = self.decoder(bottleneck, [e0, e1, e2, e3, e4])
        return output if self.training else torch.sigmoid(output)

    def attention_parameters(self):
        """Parameters trained with the lower learning rate."""
        return list(self.bottleneck_attn.parameters()) + list(self.mid_attn.parameters())


def load_weights(model, path, device="cpu"):
    """Load a notebook/train.py checkpoint dict, or a bare state_dict. Returns the dict."""
    ckpt = torch.load(path, map_location=device, weights_only=False)  # trusted checkpoint
    sd = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(sd)
    return ckpt
