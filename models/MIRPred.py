import copy
from typing import Any

import torch
from torch import nn

from .common_modules import NUM_METALS, PaperInteractionExpert, ResidueHead
from .sequence_encoder import ESMCTransformerEncoder
from .structure_encoder import GeometryCorrectedStructureEncoder


class MIRPred(nn.Module):
    """Return [batch, residues, 18] logits, optionally with training views.

    No configuration means the published checkpoint architecture. Training
    passes the complete YAML explicitly through ``from_config``.
    """

    def __init__(
        self, *, saprot_dim=1280, esmc_dim=1152, hidden_dim=128,
        vector_channels=16, radial_basis_dim=32, equivariant_layers=3,
        distance_cutoff=8.0, num_heads=8, transformer_layers=3, dropout=0.2,
        transformer_chunk=512, transformer_overlap=64,
        transformer_window_batch_size=64, classifier_hidden_dim=64,
        router_dropout=0.5, metal_router_rank=8,
    ) -> None:
        super().__init__()
        self.structure_encoder = GeometryCorrectedStructureEncoder(
            saprot_dim, hidden_dim, vector_channels, radial_basis_dim,
            equivariant_layers, distance_cutoff, dropout,
        )
        self.sequence_encoder = ESMCTransformerEncoder(
            esmc_dim, hidden_dim, num_heads, transformer_layers, dropout,
            transformer_chunk, transformer_overlap, transformer_window_batch_size,
        )
        self.anchor_head = ResidueHead(hidden_dim, classifier_hidden_dim, dropout)
        self.structure_head = ResidueHead(hidden_dim, classifier_hidden_dim, dropout)
        self.sequence_head = ResidueHead(hidden_dim, classifier_hidden_dim, dropout)
        template = PaperInteractionExpert(hidden_dim, classifier_hidden_dim, dropout)
        self.paper_experts = nn.ModuleList(copy.deepcopy(template) for _ in range(4))
        self.paper_router = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim), nn.ReLU(),
            nn.Dropout(router_dropout), nn.Linear(hidden_dim, 4),
        )
        # Match the S1 initializer and preserve the baseline RNG state.
        with torch.random.fork_rng(devices=[]):
            self.metal_router_input = nn.Linear(2 * hidden_dim, metal_router_rank)
            self.metal_router_embedding = nn.Embedding(NUM_METALS, metal_router_rank)
            self.metal_router_output = nn.Linear(metal_router_rank, 4)
            nn.init.zeros_(self.metal_router_output.weight)
            nn.init.zeros_(self.metal_router_output.bias)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "MIRPred":
        model = config["model"]
        keys = (
            "hidden_dim", "vector_channels", "radial_basis_dim",
            "equivariant_layers", "num_heads", "transformer_layers", "dropout",
            "transformer_chunk", "transformer_overlap", "transformer_window_batch_size",
            "classifier_hidden_dim", "router_dropout", "metal_router_rank",
        )
        return cls(
            **{key: model[key] for key in keys},
            saprot_dim=config["features"]["saprot_dim"],
            esmc_dim=config["features"]["esmc_dim"],
            distance_cutoff=config["graph"]["distance_cutoff"],
        )

    def forward(self, batch: dict[str, torch.Tensor], return_aux=False):
        encoded = self.structure_encoder(batch, return_aux=return_aux)
        if return_aux:
            structure, anchor, diagnostics = encoded
        else:
            structure = encoded
        sequence = self.sequence_encoder(batch["esmc"], batch["lengths"])
        full = torch.stack(
            [expert(structure, sequence) for expert in self.paper_experts], dim=-1,
        )
        joint = torch.cat((structure, sequence), dim=-1)
        residual = self.metal_router_output(
            self.metal_router_input(joint).unsqueeze(-2)
            * self.metal_router_embedding.weight[None, None, :, :]
        )
        weights = torch.softmax(self.paper_router(joint).unsqueeze(-2) + residual, dim=-1)
        weighted = (full * weights.unsqueeze(-2)).sum(dim=-1)
        logits = weighted[..., 1] - weighted[..., 0]
        if not return_aux:
            return logits
        # Ordering is intentional: auxiliary dropout and independent random
        # replacements follow the original training implementation exactly.
        auxiliary = {
            "structure": self.structure_head(structure),
            "sequence": self.sequence_head(sequence),
            "anchor": self.anchor_head(anchor),
        }
        valid = batch["residue_mask"].unsqueeze(-1) & batch["metal_mask"].unsqueeze(1)
        valid_float = valid.to(weights.dtype)
        denominator = valid_float.sum().clamp_min(1.0)
        diagnostics["router_weights"] = weights
        for index, name in enumerate(("unique_structure", "unique_sequence", "synergy", "redundancy")):
            diagnostics[f"router_{name}_mean"] = (weights[..., index] * valid_float).sum() / denominator
        without_structure = torch.stack(
            [expert(torch.randn_like(structure), sequence) for expert in self.paper_experts], dim=-1,
        )
        without_sequence = torch.stack(
            [expert(structure, torch.randn_like(sequence)) for expert in self.paper_experts], dim=-1,
        )
        diagnostics.update(
            paper_full_logits=full, paper_without_structure_logits=without_structure,
            paper_without_sequence_logits=without_sequence,
        )
        return {"logits": logits, "auxiliary_logits": auxiliary, "diagnostics": diagnostics}
