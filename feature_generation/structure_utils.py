from pathlib import Path

import numpy as np
import torch
from Bio.Align import PairwiseAligner
from Bio.PDB import MMCIFParser, PDBParser, is_aa
from scipy.spatial import cKDTree

from .prepare_pae import load_pae

AA3_TO_1 = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
    "MSE": "M",
    "SEC": "U",
    "PYL": "O",
}


def align_source_to_observed(source: str, observed: str) -> list[int | None]:
    """Return a source-position to observed-position global-alignment map."""
    if source == observed:
        return list(range(len(source)))
    aligner = PairwiseAligner(mode="global")
    aligner.match_score = 2.0
    aligner.mismatch_score = -1.0
    aligner.open_gap_score = -5.0
    aligner.extend_gap_score = -0.5
    alignment = aligner.align(source, observed)[0]
    mapping: list[int | None] = [None] * len(source)
    source_blocks, observed_blocks = alignment.aligned
    for source_block, observed_block in zip(source_blocks, observed_blocks):
        source_start, source_end = map(int, source_block)
        observed_start, observed_end = map(int, observed_block)
        mapping[source_start:source_end] = range(observed_start, observed_end)
    return mapping


def read_structure_model(path: Path):
    parser = (
        MMCIFParser(QUIET=True)
        if path.suffix.lower() in {".cif", ".mmcif"}
        else PDBParser(QUIET=True)
    )
    structure = parser.get_structure(path.stem, str(path))
    return next(structure.get_models())


def chain_residues(chain) -> list:
    return [
        residue
        for residue in chain
        if is_aa(residue, standard=False)
        or (residue.resname.upper() == "UNK" and "CA" in residue)
    ]


def resolve_structure_chain(path: Path, sequence: str) -> str:
    """Prefer an exact chain sequence, then a fully matching aligned subsegment."""
    aligned_match = None
    for chain in read_structure_model(path):
        observed = "".join(
            AA3_TO_1.get(residue.resname.upper(), "X")
            for residue in chain_residues(chain)
        )
        if observed == sequence:
            return chain.id
        if aligned_match is not None or len(observed) < len(sequence):
            continue
        mapping = align_source_to_observed(sequence, observed)
        if all(
            index is not None and aa == observed[index]
            for aa, index in zip(sequence, mapping)
        ):
            aligned_match = chain.id
    if aligned_match is not None:
        return aligned_match
    raise ValueError(f"No structure chain matches the full input sequence: {path}")


def parse_structure(path: Path, chain_id: str, sequence: str) -> tuple[np.ndarray, np.ndarray]:
    model = read_structure_model(path)
    residues = chain_residues(model[chain_id])
    observed = "".join(
        AA3_TO_1.get(residue.resname.upper(), "X") for residue in residues
    )
    mapping = align_source_to_observed(sequence, observed)
    coordinates = np.empty((len(sequence), 3), dtype=np.float32)
    plddt = np.empty(len(sequence), dtype=np.float32)
    for source_index, observed_index in enumerate(mapping):
        residue = residues[observed_index]
        if "CA" in residue:
            atom = residue["CA"]
            coordinates[source_index] = atom.coord
            plddt[source_index] = float(atom.bfactor)
        else:
            atoms = list(residue.get_atoms())
            coordinates[source_index] = np.mean([item.coord for item in atoms], axis=0)
            plddt[source_index] = float(np.mean([item.bfactor for item in atoms]))
    return coordinates, plddt


def build_edges(
    coordinates: np.ndarray, cutoff: float
) -> tuple[np.ndarray, np.ndarray]:
    length = coordinates.shape[0]
    pairs = cKDTree(coordinates).query_pairs(cutoff, output_type="ndarray")
    undirected = {tuple(map(int, pair)) for pair in pairs}
    for index in range(length - 1):
        undirected.add((index, index + 1))
    ordered = sorted(undirected)
    if not ordered:
        return np.empty((2, 0), dtype=np.int64), np.empty(0, dtype=np.float32)
    src = np.fromiter(
        (item for pair in ordered for item in (pair[0], pair[1])), dtype=np.int64
    )
    dst = np.fromiter(
        (item for pair in ordered for item in (pair[1], pair[0])), dtype=np.int64
    )
    distance = np.linalg.norm(coordinates[src] - coordinates[dst], axis=1).astype(
        np.float32
    )
    return np.stack((src, dst)), distance


def build_graph(row: dict, cutoff: float = 8.0) -> dict:
    pae_path = row.get("pae_path")
    if not row["structure_path"]:
        length = len(row["sequence"])
        return {
            "coordinates": torch.zeros((length, 3)),
            "plddt": torch.zeros(length),
            "global_plddt": torch.tensor(0.0),
            "structure_mask": torch.tensor(False),
            "edge_index": torch.empty((2, 0), dtype=torch.long),
            "edge_attr": torch.empty((0, 3)),
        }
    coordinates, plddt = parse_structure(
        Path(row["structure_path"]), row["structure_chain"], row["sequence"]
    )
    edge_index, distance = build_edges(coordinates, cutoff)
    edge_pae = np.zeros_like(distance)
    pae_mask = np.zeros_like(distance)
    if pae_path:
        pae = load_pae(Path(pae_path))
        edge_pae = pae[edge_index[0], edge_index[1]]
        pae_mask[:] = 1
    edge_attr = np.column_stack(
        (np.clip(distance / cutoff, 0, 2), np.clip(edge_pae / 31.75, 0, 1), pae_mask)
    )
    return {
        "coordinates": torch.from_numpy(coordinates),
        "plddt": torch.from_numpy(plddt),
        "global_plddt": torch.tensor(float(plddt.mean()), dtype=torch.float32),
        "structure_mask": torch.tensor(True),
        "edge_index": torch.from_numpy(edge_index).long(),
        "edge_attr": torch.from_numpy(edge_attr).float(),
    }
