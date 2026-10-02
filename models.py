"""Neural-network architectures.

Both models expect each input row to be the concatenation of two embeddings
``[A | B]`` (e.g. two 1024-d protein embeddings) and predict one logit for the
pair. Raw logits are returned; apply ``torch.sigmoid`` to get probabilities.
"""

import torch
import torch.nn as nn


class MLPClassifier(nn.Module):
    """Plain MLP on the symmetric pair features ``[A*B, |A-B|]``."""

    def __init__(self, input_dim: int = 2048, dropout1: float = 0.3, dropout2: float = 0.2):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 1024), nn.LayerNorm(1024), nn.GELU(), nn.Dropout(dropout1),
            nn.Linear(1024, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(dropout1),
            nn.Linear(512, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(dropout2),
            nn.Linear(256, 128), nn.LayerNorm(128), nn.GELU(),
            nn.Linear(128, 32), nn.LayerNorm(32), nn.GELU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = x.chunk(2, dim=1)
        pair = torch.cat((a * b, torch.abs(a - b)), dim=1)
        return self.classifier(pair)


class Siamese(nn.Module):
    """Siamese network: a shared encoder embeds A and B, then a head scores the pair."""

    def __init__(self, input_dim: int = 2048, dropout1: float = 0.3, dropout2: float = 0.2):
        super().__init__()
        embed_dim = input_dim // 2  # size of each half of the input row

        # Encoder with weights shared between both inputs.
        self.encoder = nn.Sequential(
            nn.Linear(embed_dim, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(dropout1),
            nn.Linear(512, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(dropout2),
            nn.Linear(256, 128), nn.LayerNorm(128), nn.GELU(),
        )

        # Head consumes [Ap*Bp, |Ap-Bp|] -> 2 * 128 = 256 features.
        self.head = nn.Sequential(
            nn.Linear(256, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(dropout1),
            nn.Linear(128, 64), nn.LayerNorm(64), nn.GELU(),
            nn.Linear(64, 32), nn.GELU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = x.chunk(2, dim=1)
        ap, bp = self.encoder(a), self.encoder(b)
        return self.head(torch.cat([ap * bp, torch.abs(ap - bp)], dim=1))


MODELS = {"siamese": Siamese, "mlp": MLPClassifier}


def build_model(name: str, input_dim: int) -> nn.Module:
    """Instantiate a model by name. ``input_dim`` must be even ([A | B])."""
    if name not in MODELS:
        raise ValueError(f"Unknown model '{name}'. Choose from: {sorted(MODELS)}")
    if input_dim % 2:
        raise ValueError(f"input_dim must be even (two concatenated embeddings), got {input_dim}")
    return MODELS[name](input_dim=input_dim)
