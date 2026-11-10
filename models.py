"""The two parameter-matched temperature estimators of the paper.

Conv-Pool and Conv-Attn share an identical convolutional stem and an identical
regression head and differ only in the body between them:

    Conv-Pool : three residual convolutional blocks, global average pooling
    Conv-Attn : fixed sinusoidal positional encoding, three Pre-LN Transformer
                encoder layers, mean pooling over tokens

The released checkpoints were written against these definitions. Layer names and
module order must not be changed, otherwise ``load_state_dict`` fails or, worse,
loads partially. ``check_parameter_parity`` guards against a drifted definition.
"""

import math

import torch
import torch.nn as nn

MODELS = ['Conv-Pool', 'Conv-Attn']

# trainable parameters of each model, as reported in the paper
EXPECTED_PARAMS = {'Conv-Pool': 537_857, 'Conv-Attn': 540_161}

# 4096 input pixels -> 128 tokens
STEM_DOWNSAMPLE = 32


class SharedStem(nn.Module):
    """(B, 1, 4096) -> (B, 128, 128).

    Max pooling rather than average pooling, so that an isolated high-amplitude
    pixel survives the reduction and reaches the body of either model.
    """

    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32), nn.GELU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64), nn.GELU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(128), nn.GELU(),
        )

    def forward(self, x):
        return self.stem(x)


class DeepConvBlock(nn.Module):
    """Residual block of Conv-Pool: two width-5 convolutions with batch
    normalisation. Channel-wise dropout (``Dropout1d``, p = 0.1) acts on the
    residual branch only, leaving the skip path untouched."""

    def __init__(self, dim):
        super().__init__()
        self.conv1 = nn.Conv1d(dim, dim, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(dim)
        self.conv2 = nn.Conv1d(dim, dim, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(dim)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout1d(p=0.1)

    def forward(self, x):
        res = x
        x_branch = self.gelu(self.bn1(self.conv1(x)))
        x_branch = self.bn2(self.conv2(x_branch))
        x_branch = self.dropout(x_branch)
        return self.gelu(res + x_branch)


class StellarConvPool(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = SharedStem()
        self.deep_cnn = nn.Sequential(DeepConvBlock(128), DeepConvBlock(128),
                                      DeepConvBlock(128))
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.regressor = nn.Sequential(nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, x):
        x = self.stem(x)
        x = self.deep_cnn(x)
        x = self.global_pool(x).squeeze(-1)
        return self.regressor(x)


class PositionalEncoding(nn.Module):
    """Fixed (not learned) sinusoidal positional encoding."""

    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float()
                             * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class StellarConvAttn(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = SharedStem()
        self.pos_encoder = PositionalEncoding(d_model=128)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=128, nhead=4,
            dim_feedforward=384,       # 384 rather than 512, for parameter parity
            dropout=0.1,               # attention and feed-forward dropout
            batch_first=True,
            norm_first=True,           # Pre-LN
            activation='gelu',
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3,
                                                 enable_nested_tensor=False)
        self.regressor = nn.Sequential(nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, x):
        x = self.stem(x)
        x = x.permute(0, 2, 1)         # (B, C, L) -> (B, L, C)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.regressor(x)


MODEL_REGISTRY = {'Conv-Pool': StellarConvPool, 'Conv-Attn': StellarConvAttn}


def build_model(model_name):
    return MODEL_REGISTRY[model_name]()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def check_parameter_parity(verbose=True):
    """Assert that both definitions have the parameter counts of the checkpoints."""
    counts = {m: count_parameters(build_model(m)) for m in MODELS}
    if verbose:
        p, a = counts['Conv-Pool'], counts['Conv-Attn']
        print(f'Conv-Pool : {p:,} parameters')
        print(f'Conv-Attn : {a:,} parameters')
        print(f'difference: {abs(p - a):,} ({100 * abs(p - a) / max(p, a):.2f}%)')
    assert counts == EXPECTED_PARAMS, (
        f'architecture drift: {counts} != {EXPECTED_PARAMS}; '
        'the released checkpoints will not load correctly')
    return counts
