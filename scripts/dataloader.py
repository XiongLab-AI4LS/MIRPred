import csv
import hashlib
import json
import re
from pathlib import Path

import torch

from models.common_modules import METALS
from feature_generation.structure_utils import resolve_structure_chain

_ION_SUPERSCRIPTS = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁺", "0123456789+")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_inputs(path: Path) -> list[dict]:
    rows, seen = [], set()
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, skipinitialspace=True)
        for line, raw in enumerate(reader, start=2):
            row = {
                key: (value or "").strip()
                for key, value in raw.items()
                if key is not None
            }
            protein_id = row["protein_id"]
            if protein_id in seen or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", protein_id):
                raise ValueError(f"Row {line}: protein_id must be a unique, safe file name")
            seen.add(protein_id)
            sequence = "".join(row["sequence"].split()).upper()
            row["sequence"] = sequence
            selection = row["metal ion"]
            if selection in {"", "ALL"}:
                row["metals"] = list(METALS)
            else:
                metals = [name.strip().translate(_ION_SUPERSCRIPTS) for name in selection.split(";")]
                for metal in metals:
                    if metal not in METALS:
                        raise ValueError(f"Row {line}: unsupported metal ion {metal!r}")
                row["metals"] = metals
            for key in ("structure_path", "PAE_json_path"):
                value = row.get(key, "")
                row[key] = str((path.parent / Path(value).expanduser()).resolve()) if value else ""
            row["structure_chain"] = (
                resolve_structure_chain(Path(row["structure_path"]), sequence)
                if row["structure_path"]
                else ""
            )
            rows.append(row)
    return rows


def input_fingerprint(row: dict) -> str:
    identity = {key: row[key] for key in ("protein_id", "sequence", "structure_chain")}
    identity.update(
        {
            key: file_sha256(Path(row[key])) if row[key] else ""
            for key in ("structure_path", "PAE_json_path")
        }
    )
    identity["feature_version"] = 3
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def load_feature(path: Path, row: dict, fingerprint: str) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if (
        payload.get("fingerprint") != fingerprint
        or payload.get("sequence") != row["sequence"]
    ):
        raise ValueError(f"Feature/input identity mismatch: {path}")
    return payload


def collate_proteins(samples: list[dict]) -> dict:
    lengths = torch.tensor([len(sample["sequence"]) for sample in samples])
    maximum = int(lengths.max())
    batch = {
        key: torch.zeros((len(samples), maximum, width))
        for key, width in (("saprot", 1280), ("esmc", 1152), ("coordinates", 3))
    }
    batch["plddt"] = torch.zeros((len(samples), maximum))
    edge_indices, edge_attrs, offset = [], [], 0
    for index, sample in enumerate(samples):
        length = int(lengths[index])
        for key in ("saprot", "esmc"):
            batch[key][index, :length] = sample[key].float()
        for key in ("coordinates", "plddt"):
            batch[key][index, :length] = sample["graph"][key]
        edge_indices.append(sample["graph"]["edge_index"] + offset)
        edge_attrs.append(sample["graph"]["edge_attr"])
        offset += length
    batch.update(
        lengths=lengths,
        structure_mask=torch.stack(
            [sample["graph"]["structure_mask"] for sample in samples]
        ),
        global_plddt=torch.stack(
            [sample["graph"]["global_plddt"] for sample in samples]
        ).float(),
        edge_index=torch.cat(edge_indices, dim=1),
        edge_attr=torch.cat(edge_attrs),
    )
    return batch
