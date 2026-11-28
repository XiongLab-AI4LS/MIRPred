import torch
from torch import nn


def _mean_aggregate(
    values: torch.Tensor, target: torch.Tensor, node_count: int, weights: torch.Tensor
) -> torch.Tensor:
    output = values.new_zeros((node_count, *values.shape[1:]))
    output.index_add_(0, target, values)
    degree = weights.new_zeros(node_count)
    degree.index_add_(0, target, weights)
    shape = (node_count,) + (1,) * (values.ndim - 1)
    return output / degree.clamp_min(1.0).view(shape)


class GaussianRBF(nn.Module):
    """Fixed radial basis used by the EGNN_phy geometric message layers."""

    def __init__(self, count: int, cutoff: float) -> None:
        super().__init__()
        centers = torch.linspace(0.0, float(cutoff), int(count))
        self.register_buffer("centers", centers)
        spacing = float(cutoff) / (int(count) - 1)
        self.gamma = 1.0 / max(spacing * spacing, 1e-08)

    def forward(self, distance: torch.Tensor) -> torch.Tensor:
        return torch.exp(
            -self.gamma * (distance[:, None] - self.centers[None, :]).square()
        )


class EquivariantResidueLayer(nn.Module):
    """Scalar/vector residue message passing in a shared Cartesian frame."""

    def __init__(
        self,
        hidden: int,
        vector_channels: int,
        radial_basis_dim: int,
        distance_cutoff: float,
        dropout: float,
    ) -> None:
        super().__init__()
        self.hidden = int(hidden)
        self.vector_channels = int(vector_channels)
        self.rbf = GaussianRBF(radial_basis_dim, distance_cutoff)
        edge_input = 2 * hidden + radial_basis_dim + 3
        self.edge_network = nn.Sequential(
            nn.Linear(edge_input, 2 * hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(2 * hidden, hidden + 2 * vector_channels),
        )
        self.scalar_update = nn.Sequential(
            nn.Linear(2 * hidden + vector_channels, 2 * hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(2 * hidden, hidden),
        )
        self.vector_gate = nn.Linear(hidden, vector_channels)
        self.scalar_norm = nn.LayerNorm(hidden)
        self.scalar_dropout = nn.Dropout(dropout)
        self.vector_dropout = nn.Dropout(dropout)
        self.vector_scale = nn.Parameter(torch.ones(vector_channels))

    def _normalise_vectors(self, vectors: torch.Tensor) -> torch.Tensor:
        rms = vectors.square().mean(dim=(1, 2), keepdim=True).add(1e-08).sqrt()
        return vectors / rms * self.vector_scale.view(1, 1, -1)

    def forward(
        self,
        scalar: torch.Tensor,
        vector: torch.Tensor,
        coordinates: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if edge_index.shape[1] == 0:
            invariant = vector.square().sum(dim=1).add(1e-08).sqrt()
            delta = self.scalar_update(
                torch.cat((scalar, scalar.new_zeros(scalar.shape), invariant), dim=-1)
            )
            return (self.scalar_norm(scalar + self.scalar_dropout(delta)), vector)
        (source, target) = edge_index
        displacement = coordinates[source] - coordinates[target]
        distance = displacement.norm(dim=-1).clamp_min(1e-08)
        direction = displacement / distance[:, None]
        rbf = self.rbf(distance)
        edge_state = self.edge_network(
            torch.cat((scalar[source], scalar[target], rbf, edge_attr), dim=-1)
        )
        (scalar_message, source_gate, direction_gate) = edge_state.split(
            (self.hidden, self.vector_channels, self.vector_channels), dim=-1
        )
        confidence = torch.where(
            edge_attr[:, 2] > 0,
            1.0 - edge_attr[:, 1].clamp(0.0, 1.0),
            torch.ones_like(edge_attr[:, 1]),
        )
        weights = 0.25 + 0.75 * confidence
        scalar_message = scalar_message * weights[:, None]
        vector_message = (
            vector[source] * source_gate[:, None, :]
            + direction[:, :, None] * direction_gate[:, None, :]
        ) * weights[:, None, None]
        scalar_aggregate = _mean_aggregate(
            scalar_message, target, scalar.shape[0], weights
        )
        vector_aggregate = _mean_aggregate(
            vector_message, target, scalar.shape[0], weights
        )
        invariant = vector_aggregate.square().sum(dim=1).add(1e-08).sqrt()
        scalar_delta = self.scalar_update(
            torch.cat((scalar, scalar_aggregate, invariant), dim=-1)
        )
        scalar = self.scalar_norm(scalar + self.scalar_dropout(scalar_delta))
        gate = torch.sigmoid(self.vector_gate(scalar))
        vector = vector + self.vector_dropout(gate)[:, None, :] * vector_aggregate
        vector = self._normalise_vectors(vector)
        return (scalar, vector)
