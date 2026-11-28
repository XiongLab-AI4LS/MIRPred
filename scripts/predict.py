#!/usr/bin/env python3

import argparse
import csv
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from tqdm import tqdm

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if not __package__:
    sys.path.insert(0, str(PACKAGE_ROOT))

from models.MIRPred import MIRPred
from models.common_modules import METALS
from scripts.dataloader import (
    collate_proteins,
    input_fingerprint,
    load_feature,
    read_inputs,
)
from scripts.generate_prediction_features import generate_features

CHECKPOINT_PATH = PACKAGE_ROOT / "weights" / "MIRPred_weight.pth"
# Maximum pooled validation MCC for the uncalibrated sigmoid probabilities.
PREDICTION_THRESHOLD = 0.65158932


def load_model(
    device: torch.device,
    checkpoint_path: Path = CHECKPOINT_PATH,
    threshold: float | None = None,
) -> torch.nn.Module:
    checkpoint_path = checkpoint_path.expanduser().resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint_path == CHECKPOINT_PATH.resolve():
        model = MIRPred()
        selected_threshold = PREDICTION_THRESHOLD
    else:
        if tuple(checkpoint["metals"]) != METALS or checkpoint["probability_transform"] != "sigmoid":
            raise ValueError("Expected a finalized checkpoint with the MIRPred metal order and sigmoid scores")
        model = MIRPred.from_config(checkpoint["config"])
        selected_threshold = checkpoint["prediction_threshold"]
    model.prediction_threshold = float(selected_threshold if threshold is None else threshold)
    if not 0 <= model.prediction_threshold <= 1:
        raise ValueError("Prediction threshold must lie in [0, 1]")
    model.load_state_dict(checkpoint["model"])
    return model.to(device).eval()


def predict_scores(
    rows: list[dict],
    feature_dir: Path,
    output: Path,
    device: torch.device,
    batch_size: int,
    checkpoint_path: Path = CHECKPOINT_PATH,
    threshold: float | None = None,
) -> None:
    model = load_model(device, checkpoint_path, threshold)
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as directory:
        temporary = Path(directory) / output.name
        with temporary.open(
            "w", encoding="utf-8", newline=""
        ) as handle, torch.inference_mode():
            writer = csv.writer(handle)
            writer.writerow(
                (
                    "protein_id",
                    "residue_index",
                    "amino_acid",
                    "metal ion",
                    "predicted_probability",
                    "predicted_label",
                )
            )
            for start in tqdm(range(0, len(rows), batch_size), desc="Predicting"):
                group = rows[start : start + batch_size]
                samples = [
                    load_feature(
                        feature_dir / f"{row['protein_id']}.pt",
                        row,
                        input_fingerprint(row),
                    )
                    for row in group
                ]
                batch = {
                    key: value.to(device)
                    for key, value in collate_proteins(samples).items()
                }
                scores = torch.sigmoid(model(batch).float()).cpu()
                for index, row in enumerate(group):
                    metal_columns = [(metal, METALS.index(metal)) for metal in row["metals"]]
                    for position, amino_acid in enumerate(row["sequence"]):
                        for metal, column in metal_columns:
                            score = float(scores[index, position, column])
                            writer.writerow(
                                (
                                    row["protein_id"],
                                    position + 1,
                                    amino_acid,
                                    metal,
                                    f"{score:.9g}",
                                    int(score >= model.prediction_threshold),
                                )
                            )
        temporary.replace(output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict metal-binding probabilities for residues in an input CSV."
    )
    parser.add_argument("inputs_csv", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--feature-dir", type=Path, default=PACKAGE_ROOT / "features")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument(
        "--checkpoint", type=Path, default=CHECKPOINT_PATH,
        help="Published weights or a finalized MIRPred training checkpoint",
    )
    parser.add_argument(
        "--threshold", type=float,
        help="Override the checkpoint's validation-selected probability threshold",
    )
    parser.add_argument(
        "--saprot-model",
        type=Path,
        default=PACKAGE_ROOT / "weights/pretrained/SaProt_650M_AF2",
    )
    parser.add_argument(
        "--esmc-checkpoint",
        type=Path,
        default=PACKAGE_ROOT / "weights/pretrained/esmc_600m_2024_12_v0.pth",
    )
    parser.add_argument(
        "--foldseek", type=Path, default=PACKAGE_ROOT / "tools/foldseek"
    )
    parser.add_argument("--refresh-features", action="store_true")
    arguments = parser.parse_args()
    if min(arguments.batch_size, arguments.cpu_threads) < 1:
        parser.error("--batch-size and --cpu-threads must be positive")
    return arguments


def main() -> None:
    arguments = parse_args()
    input_csv = arguments.inputs_csv.expanduser().resolve()
    output = (
        arguments.output.expanduser().resolve()
        if arguments.output
        else input_csv.parent / "predictions.csv"
    )
    if output == input_csv:
        sys.exit("Prediction output must differ from the input CSV")
    feature_dir = arguments.feature_dir.expanduser().resolve()
    device = torch.device(arguments.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("CUDA is unavailable; using CPU.")
        device = torch.device("cpu")
    torch.set_num_threads(arguments.cpu_threads)
    rows = read_inputs(input_csv)
    print(f"Preparing features for {len(rows)} protein(s).", flush=True)
    generate_features(
        rows,
        feature_dir,
        device,
        arguments.saprot_model.expanduser().resolve(),
        arguments.esmc_checkpoint.expanduser().resolve(),
        arguments.foldseek.expanduser().resolve(),
        arguments.refresh_features,
    )
    predict_scores(
        rows, feature_dir, output, device, arguments.batch_size,
        arguments.checkpoint, arguments.threshold,
    )
    print(f"Prediction output: {output}")
    print(f"Final feature directory: {feature_dir}")


if __name__ == "__main__":
    main()
