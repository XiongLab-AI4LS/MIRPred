from pathlib import Path
from types import SimpleNamespace

import torch

from .sequence_utils import chunk_starts

ESMC_DIM = 1152


def load_esmc(checkpoint: Path, device: torch.device, precision: str):
    dtype = {"float32": None, "float16": torch.float16, "bfloat16": torch.bfloat16}[precision]

    import torch.nn as nn
    from esm.layers.regression_head import RegressionHead
    from esm.layers.transformer_stack import TransformerStack
    from esm.tokenization.sequence_tokenizer import EsmSequenceTokenizer

    class EmbeddingOnlyESMC(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(64, ESMC_DIM)
            self.transformer = TransformerStack(
                ESMC_DIM, 18, None, 36, n_layers_geom=0, use_flash_attn=False
            )
            self.sequence_head = RegressionHead(ESMC_DIM, 64)
            self.tokenizer = EsmSequenceTokenizer()

        def forward(self, sequence_tokens: torch.Tensor, sequence_id: torch.Tensor):
            x = self.embed(sequence_tokens)
            chain_id = torch.ones(
                sequence_tokens.shape, dtype=torch.int64, device=x.device
            )
            for block in self.transformer.blocks:
                x = block(x, sequence_id, None, None, chain_id)
            return SimpleNamespace(embeddings=self.transformer.norm(x))

    model = EmbeddingOnlyESMC()
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    del state
    if dtype is not None:
        model = model.to(dtype=dtype)
    return model.to(device).eval(), model.tokenizer


def esmc_lookup(tokenizer) -> dict[str, int]:
    result = {}
    for amino_acid in "ACDEFGHIKLMNPQRSTVWYXBZUO":
        token, = tokenizer.encode(amino_acid, add_special_tokens=False)
        result[amino_acid] = int(token)
    return result


def _encode_esmc(
    sequences: list[str], tokenizer, lookup: dict[str, int], device: torch.device
):
    maximum = max(map(len, sequences))
    rows, masks = [], []
    for sequence in sequences:
        residues = [
            lookup.get(amino_acid, int(tokenizer.unk_token_id))
            for amino_acid in sequence
        ]
        padding = maximum - len(sequence)
        rows.append(
            [
                int(tokenizer.cls_token_id),
                *residues,
                int(tokenizer.eos_token_id),
                *([int(tokenizer.pad_token_id)] * padding),
            ]
        )
        masks.append([True] * (len(sequence) + 2) + [False] * padding)
    return torch.tensor(rows, dtype=torch.long, device=device), torch.tensor(
        masks, dtype=torch.bool, device=device
    )


def embed_esmc(
    sequence: str,
    model,
    tokenizer,
    lookup: dict[str, int],
    device: torch.device,
    maximum: int = 2046,
    overlap: int = 256,
    batch_size: int = 1,
) -> torch.Tensor:
    windows = [
        (start, min(start + maximum, len(sequence)))
        for start in chunk_starts(len(sequence), maximum, overlap)
    ]
    accumulator = torch.zeros((len(sequence), ESMC_DIM), dtype=torch.float32)
    counts = torch.zeros(len(sequence), dtype=torch.float32)
    with torch.inference_mode():
        for offset in range(0, len(windows), batch_size):
            batch = windows[offset : offset + batch_size]
            chunks = [sequence[start:end] for start, end in batch]
            tokens, mask = _encode_esmc(chunks, tokenizer, lookup, device)
            output = model(sequence_tokens=tokens, sequence_id=mask).embeddings
            for row_index, (start, end) in enumerate(batch):
                length = end - start
                accumulator[start:end] += (
                    output[row_index, 1 : length + 1].detach().float().cpu()
                )
                counts[start:end] += 1
    return accumulator / counts[:, None]
