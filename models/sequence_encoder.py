import torch
from torch import nn
from .common_modules import SinusoidalPosition


class ESMCTransformerEncoder(nn.Module):
    """ESM-C-only sequence stream with overlap-averaged Transformer windows."""

    def __init__(
        self,
        esmc_dim: int,
        hidden: int,
        heads: int,
        layers: int,
        dropout: float,
        chunk: int,
        overlap: int,
        window_batch_size: int,
    ) -> None:
        super().__init__()
        self.hidden = hidden
        self.chunk = chunk
        self.overlap = overlap
        self.window_batch_size = window_batch_size
        self.projection = nn.Sequential(
            nn.LayerNorm(esmc_dim), nn.Linear(esmc_dim, hidden), nn.GELU()
        )
        layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=heads,
            dim_feedforward=4 * hidden,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer,
            num_layers=layers,
            norm=nn.LayerNorm(hidden),
            enable_nested_tensor=False,
        )
        self.position = SinusoidalPosition(hidden)

    def _ranges(self, length: int) -> list[tuple[int, int]]:
        if length <= self.chunk:
            return [(0, length)]
        stride = self.chunk - self.overlap
        starts = list(range(0, length - self.chunk + 1, stride))
        final = length - self.chunk
        if starts[-1] != final:
            starts.append(final)
        return [(start, start + self.chunk) for start in starts]

    def forward(self, esmc: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        sequence = self.projection(esmc)
        positioned = torch.zeros_like(sequence)
        windows: list[tuple[int, int, int]] = []
        for protein_index, length_tensor in enumerate(lengths):
            length = int(length_tensor)
            current = sequence[protein_index, :length]
            positioned[protein_index, :length] = current + self.position(
                length, current.device, current.dtype
            )
            windows.extend(
                (protein_index, start, end) for start, end in self._ranges(length)
            )
        windows.sort(key=lambda item: item[2] - item[1])
        transformed = torch.zeros_like(sequence)
        counts = sequence.new_zeros((*sequence.shape[:2], 1))
        for offset in range(0, len(windows), self.window_batch_size):
            group = windows[offset : offset + self.window_batch_size]
            lengths_in_group = [end - start for _, start, end in group]
            maximum = max(lengths_in_group)
            padded = sequence.new_zeros((len(group), maximum, self.hidden))
            padding_mask = torch.ones(
                (len(group), maximum), device=sequence.device, dtype=torch.bool
            )
            for window_index, (protein_index, start, end) in enumerate(group):
                width = end - start
                padded[window_index, :width] = positioned[protein_index, start:end]
                padding_mask[window_index, :width] = False
            mask = (
                padding_mask
                if any(width != maximum for width in lengths_in_group)
                else None
            )
            encoded = self.transformer(padded, src_key_padding_mask=mask)
            for window_index, (protein_index, start, end) in enumerate(group):
                width = end - start
                transformed[protein_index, start:end] += encoded[window_index, :width]
                counts[protein_index, start:end] += 1
        return transformed / counts.clamp_min(1.0)
