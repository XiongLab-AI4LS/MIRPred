import torch
from torch import nn
from .ops import EquivariantResidueLayer


def _flatten(value: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    return torch.cat(
        [value[index, : int(length)] for (index, length) in enumerate(lengths)], dim=0
    )


def _pad(value: torch.Tensor, lengths: torch.Tensor, maximum: int) -> torch.Tensor:
    output = value.new_zeros((len(lengths), maximum, value.shape[-1]))
    offset = 0
    for index, length_tensor in enumerate(lengths):
        length = int(length_tensor)
        output[index, :length] = value[offset : offset + length]
        offset += length
    return output


class GeometryCorrectedStructureEncoder(nn.Module):
    """SaProt EGNN whose final fusion is a constrained correction expert."""

    def __init__(
        self,
        saprot_dim: int,
        hidden: int,
        vector_channels: int,
        radial_basis_dim: int,
        layers: int,
        distance_cutoff: float,
        dropout: float,
    ) -> None:
        super().__init__()
        self.vector_channels = vector_channels
        self.saprot_projection = nn.Sequential(
            nn.LayerNorm(saprot_dim), nn.Linear(saprot_dim, hidden), nn.SiLU()
        )
        self.scalar_initial = nn.Sequential(
            nn.Linear(hidden + 3, hidden), nn.LayerNorm(hidden), nn.SiLU()
        )
        self.layers = nn.ModuleList(
            (
                EquivariantResidueLayer(
                    hidden, vector_channels, radial_basis_dim, distance_cutoff, dropout
                )
                for _ in range(layers)
            )
        )
        invariant_dim = vector_channels + vector_channels * vector_channels
        evidence_dim = hidden + invariant_dim + 3
        self.evidence_projection = nn.Sequential(
            nn.Linear(evidence_dim, 2 * hidden),
            nn.LayerNorm(2 * hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(2 * hidden, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
        )
        self.error_detector = nn.Sequential(
            nn.Linear(2 * hidden + 3, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.Sigmoid(),
        )
        self.correction_generator = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
        )
        nn.init.zeros_(self.correction_generator[-1].weight)
        nn.init.zeros_(self.correction_generator[-1].bias)
        self.output_norm = nn.LayerNorm(hidden)

    def forward(self, batch: dict[str, torch.Tensor], return_aux: bool = False):
        lengths = batch["lengths"]
        maximum = batch["saprot"].shape[1]
        anchor_padded = self.saprot_projection(batch["saprot"])
        per_residue = batch["plddt"] / 100.0
        global_score = batch["global_plddt"][:, None].expand_as(per_residue) / 100.0
        structure = (
            batch["structure_mask"][:, None]
            .expand_as(per_residue)
            .to(anchor_padded.dtype)
        )
        reliability_padded = torch.stack((per_residue, global_score, structure), dim=-1)
        scalar_initial_padded = self.scalar_initial(
            torch.cat((anchor_padded, reliability_padded), dim=-1)
        )
        anchor = _flatten(anchor_padded, lengths)
        scalar_initial = _flatten(scalar_initial_padded, lengths)
        reliability = _flatten(reliability_padded, lengths)
        coordinates = _flatten(batch["coordinates"], lengths)
        scalar = scalar_initial
        vector = scalar.new_zeros((scalar.shape[0], 3, self.vector_channels))
        for layer in self.layers:
            (scalar, vector) = layer(
                scalar, vector, coordinates, batch["edge_index"], batch["edge_attr"]
            )
        vector_norm = vector.square().sum(dim=1).add(1e-08).sqrt()
        vector_gram = torch.einsum("nxc,nxd->ncd", vector, vector).flatten(1)
        evidence = self.evidence_projection(
            torch.cat(
                (scalar - scalar_initial, vector_norm, vector_gram, reliability), dim=-1
            )
        )
        gate = self.error_detector(torch.cat((anchor, evidence, reliability), dim=-1))
        reliability_gate = (
            reliability[:, 0:1] * reliability[:, 1:2] * reliability[:, 2:3]
        )
        bounded_delta = torch.tanh(self.correction_generator(evidence))
        applied_delta = gate * reliability_gate * bounded_delta
        output = self.output_norm(anchor + applied_delta)
        if return_aux:
            return _pad(output, lengths, maximum), anchor_padded, {
                "correction_gate_mean": gate.mean(),
                "correction_abs_mean": applied_delta.abs().mean(),
                "anchor_abs_mean": anchor.abs().mean(),
                "structure_reliability_mean": reliability_gate.mean(),
            }
        return _pad(output, lengths, maximum)
