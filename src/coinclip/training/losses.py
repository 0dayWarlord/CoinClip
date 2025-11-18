#loss functions for training

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    """
    contrastive loss for multimodal learning

    Encourages similar samples to have similar embeddings
    and dissimilar samples to have different embeddings
    """

    def __init__(self, temperature: float = 0.07):
        """
        initialize contrastive loss

        Args:
            temperature: temperature parameter for scaling
        """
        super().__init__()
        self.temperature = temperature

    def forward(
        self, embeddings: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """
        compute contrastive loss

        Args:
            embeddings: sample embeddings [batch_size, embedding_dim]
            labels: sample labels [batch_size]

        Returns:
            contrastive loss value
        """
        batch_size = embeddings.size(0)
        device = embeddings.device

        #normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)

        #compute similarity matrix
        similarity_matrix = torch.matmul(embeddings, embeddings.T) / self.temperature

        #create mask for positive pairs (same label)
        labels = labels.unsqueeze(1)
        mask = (labels == labels.T).float().to(device)
        mask = mask - torch.eye(batch_size, device=device)  #remove self-similarity

        #create mask for negative pairs (different labels)
        negative_mask = (labels != labels.T).float().to(device)

        #positive pairs loss
        exponential_similarity = torch.exp(similarity_matrix)
        log_probability = similarity_matrix - torch.log(exponential_similarity.sum(dim=1, keepdim=True) + 1e-8)

        #sum over positive pairs
        positive_loss = -(mask * log_probability).sum(dim=1) / (mask.sum(dim=1) + 1e-8)
        positive_loss = positive_loss.mean()

        return positive_loss


class ClassificationLoss(nn.Module):
    def __init__(self, weight: Optional[torch.Tensor] = None):
        """
        initialize classification loss

        Args:
            weight: optional class weights for imbalanced datasets
        """
        super().__init__()
        self.criterion = nn.CrossEntropyLoss(weight=weight)

    def forward(
        self, logits: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """
        compute classification loss

        Args:
            logits: model logits [batch_size, num_classes]
            labels: ground truth labels [batch_size]

        Returns:
            classification loss value
        """
        return self.criterion(logits, labels)


class CombinedLoss(nn.Module):
    """
    combined loss function

    Combines classification loss and optional contrastive loss
    """

    def __init__(
        self,
        classification_weight: float = 1.0,
        contrastive_weight: float = 0.1,
        contrastive_temperature: float = 0.07,
        class_weights: Optional[torch.Tensor] = None,
    ):
        """
        initialize combined loss

        Args:
            classification_weight: weight for classification loss
            contrastive_weight: weight for contrastive loss
            contrastive_temperature: temperature for contrastive loss
            class_weights: optional class weights for classification
        """
        super().__init__()
        self.classification_weight = classification_weight
        self.contrastive_weight = contrastive_weight

        self.classification_loss = ClassificationLoss(weight=class_weights)
        if contrastive_weight > 0:
            self.contrastive_loss = ContrastiveLoss(temperature=contrastive_temperature)
        else:
            self.contrastive_loss = None

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        embeddings: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        compute combined loss

        Args:
            logits: classification logits [batch_size, num_classes]
            labels: ground truth labels [batch_size]
            embeddings: optional embeddings for contrastive loss

        Returns:
            dictionary with 'total', 'classification', and optionally 'contrastive' losses
        """
        #classification loss
        classification_loss_value = self.classification_loss(logits, labels)
        total_loss = self.classification_weight * classification_loss_value

        losses = {
            "total": total_loss,
            "classification": classification_loss_value,
        }

        #contrastive loss (if enabled and embeddings provided)
        if self.contrastive_loss is not None and embeddings is not None:
            contrastive_loss_value = self.contrastive_loss(embeddings, labels)
            total_loss = total_loss + self.contrastive_weight * contrastive_loss_value
            losses["contrastive"] = contrastive_loss_value
            losses["total"] = total_loss

        return losses

