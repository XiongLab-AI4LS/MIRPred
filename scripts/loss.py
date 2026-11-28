from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from models.common_modules import NUM_METALS

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
AMINO_ACID_TO_INDEX = {aa: index for index, aa in enumerate(AMINO_ACIDS)}
UNKNOWN_AMINO_ACID_INDEX = len(AMINO_ACIDS)


def effective_pos_weight(raw: torch.Tensor, loss_config: dict[str, Any]) -> torch.Tensor:
    """Return the positive weight actually used by the configured BCE variant."""
    name = str(loss_config.get("name", "sqrt_capped_bce"))
    if name in {"unweighted_bce", "focal", "asymmetric"}:
        return torch.ones_like(raw)
    if name == "weighted_bce":
        return raw
    if name == "sqrt_capped_bce":
        return raw.sqrt().clamp_max(float(loss_config.get("pos_weight_cap", 10.0)))
    raise ValueError(f"Unsupported loss: {name}")


def _elementwise_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    raw_pos_weight: torch.Tensor,
    loss_config: dict[str, Any],
) -> torch.Tensor:
    name = str(loss_config.get("name", "sqrt_capped_bce"))
    if name not in {"focal", "asymmetric"}:
        return F.binary_cross_entropy_with_logits(
            logits,
            labels,
            reduction="none",
            pos_weight=effective_pos_weight(raw_pos_weight, loss_config),
        )
    probability = torch.sigmoid(logits)
    if name == "focal":
        cross_entropy = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
        gamma = float(loss_config.get("focal_gamma", 2.0))
        alpha = float(loss_config.get("focal_alpha", 0.25))
        probability_true = labels * probability + (1.0 - labels) * (1.0 - probability)
        alpha_true = labels * alpha + (1.0 - labels) * (1.0 - alpha)
        return alpha_true * (1.0 - probability_true).pow(gamma) * cross_entropy

    gamma_positive = float(loss_config.get("asymmetric_gamma_positive", 0.0))
    gamma_negative = float(loss_config.get("asymmetric_gamma_negative", 4.0))
    clip = float(loss_config.get("asymmetric_clip", 0.05))
    negative_probability = (probability - clip).clamp_min(0.0)
    positive_loss = labels * F.softplus(-logits) * (1.0 - probability).pow(gamma_positive)
    negative_loss = (1.0 - labels) * -torch.log1p(-negative_probability.clamp_max(1.0 - 1e-7))
    negative_loss = negative_loss * negative_probability.pow(gamma_negative)
    return positive_loss + negative_loss


def pooled_metal_macro_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    residue_mask: torch.Tensor,
    metal_mask: torch.Tensor,
    raw_pos_weight: torch.Tensor,
    loss_config: dict[str, Any],
) -> torch.Tensor:
    """Average residues within each metal, then equally average valid metals."""
    raw = _elementwise_loss(logits, labels, raw_pos_weight, loss_config)
    valid = residue_mask.unsqueeze(-1) & metal_mask.unsqueeze(1)
    per_metal_count = valid.sum(dim=(0, 1))
    per_metal = (raw * valid).sum(dim=(0, 1)) / per_metal_count.clamp_min(1)
    present = per_metal_count > 0
    return per_metal[present].mean() if present.any() else logits.sum() * 0.0


def pooled_masked_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    residue_mask: torch.Tensor,
    metal_mask: torch.Tensor,
    raw_pos_weight: torch.Tensor,
    loss_config: dict[str, Any],
) -> torch.Tensor:
    """Mean elementwise loss over all known metal-residue labels in a batch."""
    valid = residue_mask.unsqueeze(-1) & metal_mask.unsqueeze(1)
    raw = _elementwise_loss(logits, labels, raw_pos_weight, loss_config)
    return (raw * valid).sum() / valid.sum().clamp_min(1)


def chemical_hard_negative_ranking_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    residue_mask: torch.Tensor,
    metal_mask: torch.Tensor,
    amino_acid_indices: torch.Tensor,
    ratio: float = 5.0,
    margin: float = 0.0,
    reduction: str = "metal_macro",
    amino_acids: str = "DEHC",
) -> torch.Tensor:
    """Rank positives above the hardest valid D/E/H/C negatives for each metal."""
    chemical = torch.zeros_like(residue_mask)
    indices = tuple(AMINO_ACID_TO_INDEX[amino_acid] for amino_acid in amino_acids)
    for amino_acid_index in indices:
        chemical |= amino_acid_indices == amino_acid_index
    terms = []
    for metal_index in range(NUM_METALS):
        valid = residue_mask & metal_mask[:, metal_index, None]
        positive_logits = logits[..., metal_index][valid & labels[..., metal_index].bool()]
        negative_logits = logits[..., metal_index][
            valid & chemical & ~labels[..., metal_index].bool()
        ]
        if positive_logits.numel() == 0 or negative_logits.numel() == 0:
            continue
        wanted = max(1, int(round(float(positive_logits.numel()) * ratio)))
        hard_negative_logits = torch.topk(
            negative_logits, min(wanted, negative_logits.numel()), sorted=False
        ).values
        pair_count = max(positive_logits.numel(), hard_negative_logits.numel())
        indices = torch.arange(pair_count, device=logits.device)
        positive = positive_logits[indices % positive_logits.numel()]
        negative = hard_negative_logits[indices % hard_negative_logits.numel()]
        terms.append(F.softplus(negative - positive + margin))
    if not terms:
        return logits.sum() * 0.0
    if reduction == "pairs":
        return torch.cat(terms).mean()
    return torch.stack([term.mean() for term in terms]).mean()


def training_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    residue_mask: torch.Tensor,
    metal_mask: torch.Tensor,
    amino_acid_indices: torch.Tensor,
    raw_pos_weight: torch.Tensor,
    loss_config: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    classification_fn = (
        pooled_masked_loss if loss_config.get("reduction") == "pooled_valid_labels"
        else pooled_metal_macro_loss
    )
    classification = classification_fn(
        logits, labels, residue_mask, metal_mask, raw_pos_weight, loss_config
    )
    ranking_weight = float(loss_config.get("hard_negative_ranking_weight", 0.0))
    ranking = chemical_hard_negative_ranking_loss(
        logits,
        labels,
        residue_mask,
        metal_mask,
        amino_acid_indices,
        ratio=float(loss_config.get("hard_negative_ratio", 5.0)),
        margin=float(loss_config.get("ranking_margin", 0.0)),
        reduction=str(loss_config.get("ranking_reduction", "metal_macro")),
        amino_acids=str(loss_config.get("hard_negative_amino_acids", "DEHC")),
    ) if ranking_weight > 0 else logits.sum() * 0.0
    total = classification + ranking_weight * ranking
    return total, {"classification": classification.detach(), "ranking": ranking.detach()}


def probability_interaction_regularization(
    diagnostics: dict[str, torch.Tensor],
    residue_mask: torch.Tensor,
    metal_mask: torch.Tensor,
    *, unique_margin: float = 0.1,
    synergy_margin: float = 0.1,
    reduction: str = "pooled_valid_labels",
    labels: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compare the three views using absolute differences in expert probabilities."""
    unique_margin = float(unique_margin)
    synergy_margin = float(synergy_margin)
    valid = residue_mask.unsqueeze(-1) & metal_mask.unsqueeze(1)
    names = (
        "interaction_unique_structure", "interaction_unique_sequence",
        "interaction_synergy", "interaction_redundancy",
    )

    def probabilities(key: str) -> torch.Tensor:
        # Evaluate the class probability in FP32, including under AMP.
        logits = diagnostics[key][valid].float()
        return torch.sigmoid(logits[:, 1, :] - logits[:, 0, :])

    full = probabilities("paper_full_logits")
    if full.shape[0] == 0:
        zero = full.sum()
        return zero, {name: zero.detach() for name in names}
    no_structure = probabilities("paper_without_structure_logits")
    no_sequence = probabilities("paper_without_sequence_logits")
    distance_no_structure = (full - no_structure).abs()
    distance_no_sequence = (full - no_sequence).abs()

    def reduce(values: torch.Tensor) -> torch.Tensor:
        if reduction == "pooled_valid_labels":
            return values.mean()
        if reduction == "class_balanced":
            selected_labels = labels[valid]
            groups = [selected_labels == category for category in (0, 1)]
        else:
            metal_indices = valid.nonzero(as_tuple=True)[2]
            groups = [metal_indices == index for index in range(valid.shape[-1])]
        present = [values[group].mean() for group in groups if group.any()]
        return torch.stack(present).mean() if present else values.sum() * 0.0

    unique_structure = F.relu(
        distance_no_sequence[:, 0] - distance_no_structure[:, 0] + unique_margin
    )
    unique_structure = reduce(unique_structure)
    unique_sequence = F.relu(
        distance_no_structure[:, 1] - distance_no_sequence[:, 1] + unique_margin
    )
    unique_sequence = reduce(unique_sequence)
    synergy = 0.5 * (
        reduce(F.relu(synergy_margin - distance_no_structure[:, 2]))
        + reduce(F.relu(synergy_margin - distance_no_sequence[:, 2]))
    )
    redundancy = 0.5 * (
        reduce(distance_no_structure[:, 3]) + reduce(distance_no_sequence[:, 3])
    )
    terms = (unique_structure, unique_sequence, synergy, redundancy)
    total = torch.stack(terms).mean()
    return total, {name: term.detach() for name, term in zip(names, terms)}


def _macro_masked_mean(
    values: torch.Tensor, residue_mask: torch.Tensor, metal_mask: torch.Tensor
) -> torch.Tensor:
    valid = residue_mask.unsqueeze(-1) & metal_mask.unsqueeze(1)
    count = valid.sum(dim=(0, 1))
    per_metal = (values * valid).sum(dim=(0, 1)) / count.clamp_min(1)
    present = count > 0
    return per_metal[present].mean() if present.any() else values.sum() * 0.0


def _pooled_masked_mean(
    values: torch.Tensor, residue_mask: torch.Tensor, metal_mask: torch.Tensor
) -> torch.Tensor:
    valid = residue_mask.unsqueeze(-1) & metal_mask.unsqueeze(1)
    return (values * valid).sum() / valid.sum().clamp_min(1)


def _correction_loss(
    corrected_logits: torch.Tensor,
    anchor_logits: torch.Tensor,
    labels: torch.Tensor,
    residue_mask: torch.Tensor,
    metal_mask: torch.Tensor,
    raw_pos_weight: torch.Tensor,
    loss_config: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Focus correction on anchor errors and preserve already-correct semantics."""
    pos_weight = effective_pos_weight(raw_pos_weight, loss_config)
    error = (labels - torch.sigmoid(anchor_logits.detach())).abs()
    elementwise = F.binary_cross_entropy_with_logits(
        corrected_logits, labels, reduction="none", pos_weight=pos_weight
    )
    reduce_mean = _pooled_masked_mean if loss_config.get("reduction") == "pooled_valid_labels" else _macro_masked_mean
    error_focused = reduce_mean(
        elementwise * (0.25 + error), residue_mask, metal_mask
    )
    probability_shift = (
        torch.sigmoid(corrected_logits) - torch.sigmoid(anchor_logits.detach())
    ).abs()
    no_harm = reduce_mean(
        probability_shift * (1.0 - error), residue_mask, metal_mask
    )
    total = error_focused + float(loss_config.get("correction_no_harm_weight", 0.1)) * no_harm
    return total, {
        "correction_error_focused": error_focused.detach(),
        "correction_no_harm": no_harm.detach(),
    }


def multimodal_training_loss(
    output: dict[str, Any],
    batch: dict[str, torch.Tensor],
    raw_pos_weight: torch.Tensor,
    config: dict[str, Any],
    epoch: int,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    loss_config = config["training"]["loss"]
    auxiliary = config["training"].get("auxiliary", {})
    logits = output["logits"]
    total, parts = training_loss(
        logits,
        batch["labels"],
        batch["residue_mask"],
        batch["metal_mask"],
        batch["amino_acid_indices"],
        raw_pos_weight,
        loss_config,
    )
    variant = str(config["model"]["variant"])
    classify = (
        pooled_masked_loss if loss_config.get("reduction") == "pooled_valid_labels"
        else pooled_metal_macro_loss
    )
    expert_weight = float(auxiliary.get("expert_weight", 0.0))
    expert_terms = []
    corrected_fusion = variant in {
        "dense_fusion",
        "four_expert",
        "four_expert_interaction",
        "four_expert_dense_residual",
        "four_expert_router_calibrated",
        "four_expert_deep_synergy",
        "four_expert_paper_correction",
    }
    if corrected_fusion or variant == "dense_fusion_anchor_only":
        for name, branch_logits in output["auxiliary_logits"].items():
            if name == "anchor":
                continue
            expert_terms.append(
                classify(
                    branch_logits,
                    batch["labels"],
                    batch["residue_mask"],
                    batch["metal_mask"],
                    raw_pos_weight,
                    loss_config,
                )
            )
    expert_loss = (
        torch.stack(expert_terms).mean() if expert_terms else logits.sum() * 0.0
    )
    total = total + expert_weight * expert_loss
    parts["expert"] = expert_loss.detach()

    anchor_weight = float(auxiliary.get("anchor_weight", 0.0))
    anchor_loss = logits.sum() * 0.0
    if anchor_weight > 0:
        anchor_logits = output["auxiliary_logits"]["anchor"]
        anchor_loss = classify(
            anchor_logits, batch["labels"], batch["residue_mask"],
            batch["metal_mask"], raw_pos_weight, loss_config,
        )
        total = total + anchor_weight * anchor_loss
    parts["anchor"] = anchor_loss.detach()

    correction_weight = float(auxiliary.get("correction_weight", 0.0))
    correction_loss = logits.sum() * 0.0
    correction_parts = {
        "correction_error_focused": correction_loss.detach(),
        "correction_no_harm": correction_loss.detach(),
    }
    if corrected_fusion or variant == "egnn_corrected_error_aware":
        correction_loss, correction_parts = _correction_loss(
            output["auxiliary_logits"]["structure"],
            output["auxiliary_logits"]["anchor"],
            batch["labels"],
            batch["residue_mask"],
            batch["metal_mask"],
            raw_pos_weight,
            loss_config,
        )
        total = total + correction_weight * correction_loss
    parts["correction"] = correction_loss.detach()
    parts.update(correction_parts)

    interaction_weight = float(auxiliary.get("interaction_weight", 0.0))
    interaction = config["training"]["interaction"]
    interaction_loss, interaction_parts = probability_interaction_regularization(
        output["diagnostics"], batch["residue_mask"], batch["metal_mask"],
        unique_margin=interaction["unique_margin"],
        synergy_margin=interaction["synergy_margin"],
        reduction=interaction.get("reduction", "pooled_valid_labels"),
        labels=batch["labels"],
    )
    warmup_epochs = int(auxiliary.get("interaction_warmup_epochs", 3))
    interaction_scale = 1.0 if warmup_epochs <= 0 else min(1.0, float(epoch) / warmup_epochs)
    total = total + interaction_weight * interaction_scale * interaction_loss
    parts["interaction"] = interaction_loss.detach()
    parts.update(interaction_parts)
    for name in (
        "shared_invariance", "synergy_dependence", "interaction_unique_structure",
        "interaction_unique_sequence", "interaction_synergy", "interaction_redundancy",
    ):
        parts.setdefault(name, logits.sum().detach() * 0.0)

    router_loss = logits.sum() * 0.0
    router_keys = [
        f"router_{name}_mean"
        for name in ("structure", "sequence", "shared", "synergy")
    ]
    if "router_weights" in output["diagnostics"]:
        weights = output["diagnostics"]["router_weights"]
        valid = batch["residue_mask"].unsqueeze(-1) & batch["metal_mask"].unsqueeze(1)
        valid_float = valid.to(weights.dtype)
        per_metal_count = valid_float.sum(dim=(0, 1))
        per_metal_mean = (
            weights * valid_float.unsqueeze(-1)
        ).sum(dim=(0, 1)) / per_metal_count.clamp_min(1.0).unsqueeze(-1)
        present = per_metal_count > 0
        router_loss = (per_metal_mean[present] - 0.25).square().mean()
        total = total + float(auxiliary.get("router_balance_weight", 0.0)) * router_loss
    elif all(key in output["diagnostics"] for key in router_keys):
        router_mean = torch.stack([output["diagnostics"][key] for key in router_keys])
        router_loss = (router_mean - 0.25).square().mean()
        total = total + float(auxiliary.get("router_balance_weight", 0.0)) * router_loss
    parts["router_balance"] = router_loss.detach()
    for key, value in output["diagnostics"].items():
        if value.ndim == 0:
            parts[key] = value.detach()
    parts["total"] = total.detach()
    return total, parts
