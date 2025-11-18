#evaluation metrics

from typing import Dict, List, Optional

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_classification_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int = 2,
    average: str = "binary",
) -> Dict[str, float]:
    """
    compute classification metrics

    Args:
        logits: model logits [batch_size, num_classes]
        labels: ground truth labels [batch_size]
        num_classes: number of classes
        average: averaging strategy for multi-class ('binary', 'macro', 'micro', 'weighted')

    Returns:
        dictionary of metrics
    """
    #convert to numpy
    if isinstance(logits, torch.Tensor):
        logits = logits.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()

    #get predictions
    predictions = np.argmax(logits, axis=1)
    probabilities = torch.softmax(torch.tensor(logits), dim=1).numpy()

    metrics = {}

    #accuracy
    metrics["accuracy"] = accuracy_score(labels, predictions)

    #precision, Recall, F1
    if num_classes == 2:
        metrics["precision"] = precision_score(labels, predictions, average=average, zero_division=0)
        metrics["recall"] = recall_score(labels, predictions, average=average, zero_division=0)
        metrics["f1"] = f1_score(labels, predictions, average=average, zero_division=0)

        #ROC-AUC and PR-AUC
        try:
            metrics["roc_auc"] = roc_auc_score(labels, probabilities[:, 1])
        except ValueError:
            metrics["roc_auc"] = 0.0

        try:
            metrics["pr_auc"] = average_precision_score(labels, probabilities[:, 1])
        except ValueError:
            metrics["pr_auc"] = 0.0
    else:
        #multi-class metrics
        metrics["precision"] = precision_score(
            labels, predictions, average=average, zero_division=0
        )
        metrics["recall"] = recall_score(labels, predictions, average=average, zero_division=0)
        metrics["f1"] = f1_score(labels, predictions, average=average, zero_division=0)

        #multi-class ROC-AUC (one-vs-rest)
        try:
            metrics["roc_auc"] = roc_auc_score(
                labels, probabilities, multi_class="ovr", average=average
            )
        except ValueError:
            metrics["roc_auc"] = 0.0

    return metrics


def compute_ranking_metrics(
    scores: torch.Tensor,
    labels: torch.Tensor,
    k: Optional[int] = None,
) -> Dict[str, float]:
    """
    compute ranking metrics (e.g., NDCG, MRR)

    Args:
        scores: ranking scores [batch_size]
        labels: relevance labels [batch_size]
        k: optional cutoff for top-k metrics

    Returns:
        dictionary of ranking metrics
    """
    if isinstance(scores, torch.Tensor):
        scores = scores.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()

    #sort by scores
    sorted_indices = np.argsort(scores)[::-1]
    sorted_labels = labels[sorted_indices]

    metrics = {}

    #mean Reciprocal Rank (MRR)
    first_relevant = np.where(sorted_labels > 0)[0]
    if len(first_relevant) > 0:
        metrics["mrr"] = 1.0 / (first_relevant[0] + 1)
    else:
        metrics["mrr"] = 0.0

    #NDCG@k
    if k is not None:
        k = min(k, len(sorted_labels))
        discounted_cumulative_gain = np.sum(sorted_labels[:k] / np.log2(np.arange(2, k + 2)))
        ideal_labels = np.sort(labels)[::-1]
        ideal_discounted_cumulative_gain = np.sum(ideal_labels[:k] / np.log2(np.arange(2, k + 2)))
        metrics[f"ndcg@{k}"] = discounted_cumulative_gain / (ideal_discounted_cumulative_gain + 1e-8)
    else:
        #NDCG (full)
        discounted_cumulative_gain = np.sum(sorted_labels / np.log2(np.arange(2, len(sorted_labels) + 2)))
        ideal_labels = np.sort(labels)[::-1]
        ideal_discounted_cumulative_gain = np.sum(ideal_labels / np.log2(np.arange(2, len(ideal_labels) + 2)))
        metrics["ndcg"] = discounted_cumulative_gain / (ideal_discounted_cumulative_gain + 1e-8)

    return metrics

