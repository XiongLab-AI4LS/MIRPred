import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np


def load_pae(path: Path) -> np.ndarray:
    with np.load(path) as data:
        return data["pae"].astype(np.float32)


def prepare_pae(row: dict, cache_dir: Path) -> str:
    """Return a cached NPZ path, or an empty string when JSON was not supplied."""
    if not row["PAE_json_path"]:
        return ""
    source = Path(row["PAE_json_path"])
    raw = source.read_bytes()
    source_hash = hashlib.sha256(raw).hexdigest()
    sequence = row["sequence"]
    sequence_hash = hashlib.sha256(sequence.encode("ascii")).hexdigest()
    destination = cache_dir / f"{row['protein_id']}.npz"
    if destination.is_file():
        with np.load(destination) as data:
            if (
                data["source_json_sha256"].item() == source_hash
                and data["sequence_sha256"].item() == sequence_hash
            ):
                return str(destination)

    payload = json.loads(raw)
    if isinstance(payload, list):
        payload, = payload
    key = "predicted_aligned_error" if "predicted_aligned_error" in payload else "pae"
    matrix = np.asarray(payload[key], dtype=np.float32)
    if matrix.shape != (len(sequence), len(sequence)):
        raise ValueError(f"{source}: expected an L x L PAE matrix")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=destination.parent) as directory:
        temporary = Path(directory) / destination.name
        with temporary.open("wb") as handle:
            np.savez_compressed(
                handle,
                pae=matrix.astype(np.float16),
                sequence_sha256=sequence_hash,
                source_json_sha256=source_hash,
            )
        temporary.replace(destination)
    return str(destination)
