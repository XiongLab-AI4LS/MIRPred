import math
import torch
from torch import nn

METALS = (
    "Zn2+",
    "Ca2+",
    "Mg2+",
    "Mn2+",
    "Fe_cation",
    "Cu_cation",
    "K+",
    "Ni2+",
    "Na+",
    "Fe3+",
    "Cu2+",
    "Fe2+",
    "Cd2+",
    "Cu+",
    "Co2+",
    "Hg2+",
    "Co3+",
    "Li+",
)
NUM_METALS = len(METALS)


class SinusoidalPosition(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(
        self, length: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        position = torch.arange(length, device=device, dtype=torch.float32)[:, None]
        scale = torch.exp(
            torch.arange(0, self.dim, 2, device=device, dtype=torch.float32)
            * (-math.log(10000.0) / self.dim)
        )
        result = torch.zeros((length, self.dim), device=device, dtype=torch.float32)
        result[:, 0::2] = torch.sin(position * scale)
        result[:, 1::2] = torch.cos(position * scale[: result[:, 1::2].shape[1]])
        return result.to(dtype)


class ResidueHead(nn.Module):
    def __init__(self, input_dim: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, NUM_METALS),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.network(value)


class PaperInteractionExpert(nn.Module):
    """Independent multimodal fusion expert with two logits per metal."""

    def __init__(self, hidden: int, classifier_hidden: int, dropout: float) -> None:
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Linear(4 * hidden, 4 * hidden),
            nn.LayerNorm(4 * hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden, classifier_hidden),
            nn.LayerNorm(classifier_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden, NUM_METALS * 2),
        )

    def forward(self, structure: torch.Tensor, sequence: torch.Tensor) -> torch.Tensor:
        features = torch.cat(
            (structure, sequence, structure * sequence, (structure - sequence).abs()),
            dim=-1,
        )
        return self.head(self.fusion(features)).reshape(
            *structure.shape[:2], NUM_METALS, 2
        )
