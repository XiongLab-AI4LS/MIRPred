import gc
import tempfile
from pathlib import Path

import torch
from tqdm import tqdm

from feature_generation.assemble_model_features import save_features
from feature_generation.prepare_pae import prepare_pae
from feature_generation.structure_utils import build_graph
from scripts.dataloader import file_sha256, input_fingerprint, load_feature


def generate_features(
    rows: list[dict],
    feature_dir: Path,
    device: torch.device,
    saprot_model: Path,
    esmc_checkpoint: Path,
    foldseek: Path,
    refresh: bool = False,
) -> None:
    rows = [dict(row, pae_path=prepare_pae(row, feature_dir / "pae")) for row in rows]
    fingerprints = {row["protein_id"]: input_fingerprint(row) for row in rows}
    sources = {"saprot": saprot_model / "pytorch_model.bin", "esmc": esmc_checkpoint}
    source_hashes = {
        key: file_sha256(path) for key, path in sources.items() if path.is_file()
    }
    pending = rows if refresh else []
    if not refresh:
        for row in rows:
            protein_id = row["protein_id"]
            path = feature_dir / f"{protein_id}.pt"
            try:
                payload = load_feature(path, row, fingerprints[protein_id])
            except (FileNotFoundError, ValueError):
                pending.append(row)
                continue
            cached = payload.get("source_models", {})
            if any(cached.get(key) != value for key, value in source_hashes.items()):
                pending.append(row)
    if not pending:
        return
    from feature_generation.extract_saprot_features import (
        build_saprot_descriptor,
        embed_saprot,
        load_saprot,
    )
    from feature_generation.extract_sequence_embeddings import (
        embed_esmc,
        esmc_lookup,
        load_esmc,
    )

    with tempfile.TemporaryDirectory(prefix="mirpred_features_") as temporary:
        workspace = Path(temporary)
        saprot_precision = "float16" if device.type == "cuda" else "float32"
        model, tokenizer = load_saprot(saprot_model, device, saprot_precision)
        for row in tqdm(pending, desc="SaProt features"):
            descriptor = build_saprot_descriptor(row, foldseek)
            embedding = embed_saprot(descriptor, model, tokenizer, device)
            torch.save(embedding, workspace / f"{row['protein_id']}.saprot.pt")
        del model, tokenizer
        gc.collect()
        esmc_precision = "bfloat16" if device.type == "cuda" else "float32"
        model, tokenizer = load_esmc(esmc_checkpoint, device, esmc_precision)
        lookup = esmc_lookup(tokenizer)
        for row in tqdm(pending, desc="ESM-C features"):
            esmc = embed_esmc(row["sequence"], model, tokenizer, lookup, device)
            saprot = torch.load(
                workspace / f"{row['protein_id']}.saprot.pt",
                map_location="cpu",
                weights_only=True,
            )
            save_features(
                feature_dir / f"{row['protein_id']}.pt",
                row,
                fingerprints[row["protein_id"]],
                saprot,
                esmc,
                build_graph(row),
                source_hashes,
            )
