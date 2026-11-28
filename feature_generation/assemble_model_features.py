from pathlib import Path
from tempfile import TemporaryDirectory

import torch


def save_features(
    path: Path,
    row: dict,
    fingerprint: str,
    saprot: torch.Tensor,
    esmc: torch.Tensor,
    graph: dict,
    sources: dict,
) -> None:
    payload = {
        "fingerprint": fingerprint,
        "sequence": row["sequence"],
        "saprot": saprot.to(torch.float16),
        "esmc": esmc.to(torch.float16),
        "graph": graph,
        "source_models": sources,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=path.parent) as directory:
        temporary = Path(directory) / path.name
        torch.save(payload, temporary)
        temporary.replace(path)
