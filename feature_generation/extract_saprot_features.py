import json
import subprocess
import tempfile
from pathlib import Path

import torch
from transformers import EsmConfig, EsmForMaskedLM, EsmTokenizer

from .sequence_utils import chunk_starts
from .structure_utils import align_source_to_observed

SAPROT_DIM = 1280
STANDARD_AA = frozenset("ACDEFGHIKLMNPQRSTVWY")


def _descriptor_chain(name: str, structure_name: str) -> str:
    token = name.split(" ", 1)[0]
    prefix = structure_name + "_"
    return (
        token[len(prefix) :] if token.startswith(prefix) else token.rsplit("_", 1)[-1]
    )


def build_saprot_descriptor(row: dict[str, str], foldseek: Path) -> dict[str, str]:
    sequence = row["sequence"]
    amino_tokens = "".join(aa if aa in STANDARD_AA else "#" for aa in sequence)
    structure_tokens = "#" * len(sequence)
    if row["structure_path"]:
        structure_tokens = _foldseek_tokens(row, foldseek)
    return {
        "saprot_amino_acid_sequence": amino_tokens,
        "saprot_structure_sequence": structure_tokens,
    }


def _foldseek_tokens(row: dict[str, str], foldseek: Path) -> str:
    sequence = row["sequence"]
    structure_path = Path(row["structure_path"])
    protein_id = row["protein_id"]
    with tempfile.TemporaryDirectory(prefix="mirpred_foldseek_") as directory:
        output = Path(directory) / "descriptor.tsv"
        command = [
            str(foldseek), "structureto3didescriptor", "-v", "0",
            "--threads", "1", "--chain-name-mode", "1",
            str(structure_path), str(output),
        ]
        subprocess.run(command, check=True, timeout=1800)
        records = output.read_text(encoding="utf-8").splitlines()

    candidates = []
    for record in records:
        fields = record.split("\t")
        if len(fields) < 3:
            continue
        if _descriptor_chain(fields[0], structure_path.name) != row["structure_chain"]:
            continue
        observed, tokens = fields[1:3]
        mapping = align_source_to_observed(sequence, observed)
        matches = sum(
            j is not None and aa == observed[j] for aa, j in zip(sequence, mapping)
        )
        score = matches, sum(j is not None for j in mapping)
        candidates.append((score, observed, tokens, mapping))

    _, observed, tokens, mapping = max(candidates, key=lambda candidate: candidate[0])
    permitted = set(json.loads(row.get("nonstandard_structure_positions", "[]")))
    for i, (aa, j) in enumerate(zip(sequence, mapping)):
        if j is not None and aa == observed[j]:
            continue
        if i in permitted and aa in "UO" and (j is None or observed[j] == "X"):
            continue
        raise ValueError(f"{protein_id}: Foldseek sequence mismatch at residue {i + 1}")
    return "".join("#" if j is None else tokens[j] for j in mapping)


def load_saprot(model_directory: Path, device: torch.device, precision: str):
    dtype = {"float32": None, "float16": torch.float16}[precision]
    tokenizer = EsmTokenizer.from_pretrained(
        str(model_directory), local_files_only=True
    )
    configuration = EsmConfig.from_pretrained(
        str(model_directory), local_files_only=True
    )
    masked_model = EsmForMaskedLM(configuration)
    checkpoint = model_directory / "pytorch_model.bin"
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = state.get("state_dict", state)
    defaults = masked_model.state_dict()
    if "esm.embeddings.position_ids" not in defaults:
        state.pop("esm.embeddings.position_ids", None)
    for key in (
        "esm.embeddings.position_embeddings.weight",
        "esm.contact_head.regression.weight",
        "esm.contact_head.regression.bias",
    ):
        if key in defaults:
            state.setdefault(key, defaults[key])
    masked_model.load_state_dict(state)
    model = masked_model.esm
    del masked_model, state
    if dtype is not None:
        model = model.to(dtype=dtype)
    return model.to(device).eval(), tokenizer


def embed_saprot(
    descriptor: dict[str, str],
    model,
    tokenizer,
    device: torch.device,
    maximum: int = 1024,
    overlap: int = 128,
    batch_size: int = 1,
) -> torch.Tensor:
    amino = descriptor["saprot_amino_acid_sequence"]
    structure = descriptor["saprot_structure_sequence"]
    windows = []
    for start in chunk_starts(len(amino), maximum, overlap):
        end = min(start + maximum, len(amino))
        token_string = "".join(
            a + b.lower() for a, b in zip(amino[start:end], structure[start:end])
        )
        windows.append((start, end, token_string))
    accumulator = torch.zeros((len(amino), SAPROT_DIM), dtype=torch.float32)
    counts = torch.zeros(len(amino), dtype=torch.float32)
    with torch.inference_mode():
        for offset in range(0, len(windows), batch_size):
            batch = windows[offset : offset + batch_size]
            encoded = tokenizer(
                [item[2] for item in batch], return_tensors="pt", padding=True
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            output = model(**encoded).last_hidden_state
            for row_index, (start, end, _) in enumerate(batch):
                length = end - start
                accumulator[start:end] += (
                    output[row_index, 1 : length + 1].detach().float().cpu()
                )
                counts[start:end] += 1
    return accumulator / counts[:, None]
