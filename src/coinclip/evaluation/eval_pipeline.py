#evaluation pipeline

import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from coinclip.models.coinclip_model import CoinCLIPModel
from coinclip.training.losses import CombinedLoss
from coinclip.training.metrics import compute_classification_metrics, compute_ranking_metrics
from coinclip.utils.checkpointing import load_checkpoint
from coinclip.utils.logging import setup_logger


def evaluate_model(
    model: CoinCLIPModel,
    data_loader: DataLoader,
    device: str = "cuda",
    logger=None,
    return_predictions: bool = False,
) -> Dict[str, float]:
    """
    evaluate model on a dataset

    Args:
        model: model to evaluate
        data_loader: data loader for evaluation
        device: device to evaluate on
        logger: optional logger
        return_predictions: whether to return predictions

    Returns:
        dictionary of evaluation metrics
    """
    if logger is None:
        logger = setup_logger()

    model = model.to(device)
    model.eval()

    #initialize loss function
    loss_function = CombinedLoss(
        classification_weight=1.0,
        contrastive_weight=0.1 if model.use_contrastive else 0.0,
        contrastive_temperature=model.contrastive_temperature if model.use_contrastive else 0.07,
    )

    total_loss = 0.0
    all_logits = []
    all_labels = []
    all_predictions = []
    all_sample_ids = []

    with torch.no_grad():
        evaluation_progress_bar = tqdm(data_loader, desc="Evaluating")
        for batch in evaluation_progress_bar:
            #move to device
            text = {k: v.to(device) for k, v in batch["text"].items()}
            image = batch["image"].to(device)
            numeric = batch["numeric"].to(device)
            labels = batch["label"].to(device)

            #forward pass
            outputs = model(text=text, image=image, numeric=numeric)

            #compute loss
            embeddings = outputs.get("contrastive_emb") if model.use_contrastive else None
            losses = loss_function(
                logits=outputs["logits"],
                labels=labels,
                embeddings=embeddings,
            )

            #accumulate
            total_loss += losses["total"].item()
            all_logits.append(outputs["logits"])
            all_labels.append(labels)
            all_predictions.append(torch.argmax(outputs["logits"], dim=1))
            all_sample_ids.append(batch.get("sample_id", torch.zeros(len(labels))))

    #concatenate all results
    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    all_predictions = torch.cat(all_predictions, dim=0)
    all_sample_ids = torch.cat(all_sample_ids, dim=0)

    #compute metrics
    average_loss = total_loss / len(data_loader)
    metrics = compute_classification_metrics(
        all_logits, all_labels, num_classes=model.classifier[-1].out_features
    )
    metrics["loss"] = average_loss

    #log results
    logger.info("Evaluation Results:")
    for key, value in metrics.items():
        logger.info(f"  {key}: {value:.4f}")

    result = {"metrics": metrics}

    if return_predictions:
        result["predictions"] = all_predictions.cpu().numpy()
        result["labels"] = all_labels.cpu().numpy()
        result["sample_ids"] = all_sample_ids.cpu().numpy()
        result["logits"] = all_logits.cpu().numpy()

    return result


def save_evaluation_results(
    results: Dict,
    output_path: Path,
) -> None:
    """
    save evaluation results to JSON file

    Args:
        results: evaluation results dictionary
        output_path: path to save results
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    #convert numpy arrays to lists for JSON serialization
    serializable_results = {}
    for key, value in results.items():
        if key == "metrics":
            serializable_results[key] = value
        else:
            if isinstance(value, torch.Tensor):
                value = value.cpu().numpy()
            if isinstance(value, np.ndarray):
                serializable_results[key] = value.tolist()
            else:
                serializable_results[key] = value

    with open(output_path, "w") as f:
        json.dump(serializable_results, f, indent=2)

