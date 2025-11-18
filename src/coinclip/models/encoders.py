#modality-specific encoders

from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoModel


class TextEncoder(nn.Module):
    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        embedding_dim: int = 768,
        output_dim: Optional[int] = None,
        freeze_backbone: bool = False,
    ):
        """
        initialize text encoder

        Args:
            model_name: huggingFace model name
            embedding_dim: dimension of transformer output
            output_dim: optional projection dimension
            freeze_backbone: whether to freeze transformer weights
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.transformer = AutoModel.from_pretrained(model_name)

        if freeze_backbone:
            for parameter in self.transformer.parameters():
                parameter.requires_grad = False

        #projection layer if output_dim is specified
        self.output_dim = output_dim or embedding_dim
        if output_dim and output_dim != embedding_dim:
            self.projection = nn.Linear(embedding_dim, output_dim)
        else:
            self.projection = nn.Identity()

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        encode text

        Args:
            input_ids: token IDs [batch_size, seq_len]
            attention_mask: attention mask [batch_size, seq_len]

        Returns:
            text embeddings [batch_size, output_dim]
        """
        outputs = self.transformer(
            input_ids=input_ids, attention_mask=attention_mask
        )
        #use [CLS] token embedding
        pooled_output = outputs.last_hidden_state[:, 0, :]
        return self.projection(pooled_output)


class ImageEncoder(nn.Module):
    def __init__(
        self,
        model_name: str = "resnet50",
        embedding_dim: int = 2048,
        output_dim: Optional[int] = None,
        pretrained: bool = True,
        freeze_backbone: bool = False,
    ):
        """
        initialize image encoder

        Args:
            model_name: model architecture ('resnet50', 'vit', etc.)
            embedding_dim: dimension of backbone output
            output_dim: optional projection dimension
            pretrained: whether to use pretrained weights
            freeze_backbone: whether to freeze backbone weights
        """
        super().__init__()
        self.embedding_dim = embedding_dim

        if model_name == "resnet50":
            import torchvision.models as models

            backbone = models.resnet50(pretrained=pretrained)
            #remove final classification layer
            self.backbone = nn.Sequential(*list(backbone.children())[:-1])
            self.embedding_dim = 2048
        elif model_name == "vit":
            from transformers import ViTModel

            self.backbone = ViTModel.from_pretrained(
                "google/vit-base-patch16-224" if pretrained else None
            )
            self.embedding_dim = 768
        else:
            raise ValueError(f"Unsupported model: {model_name}")

        if freeze_backbone:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False

        #projection layer
        self.output_dim = output_dim or self.embedding_dim
        if output_dim and output_dim != self.embedding_dim:
            self.projection = nn.Linear(self.embedding_dim, output_dim)
        else:
            self.projection = nn.Identity()

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        encode images

        Args:
            images: image tensor [batch_size, 3, H, W]

        Returns:
            image embeddings [batch_size, output_dim]
        """
        if isinstance(self.backbone, nn.Sequential):
            #resNet
            features = self.backbone(images)
            features = features.view(features.size(0), -1)
        else:
            #viT
            outputs = self.backbone(pixel_values=images)
            features = outputs.last_hidden_state[:, 0, :]  #[CLS] token

        return self.projection(features)


class NumericEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int = 6,
        hidden_dims: list[int] = [128, 64],
        output_dim: int = 64,
        dropout: float = 0.1,
    ):
        """
        initialize numeric encoder

        Args:
            input_dim: input feature dimension
            hidden_dims: hidden layer dimensions
            output_dim: output embedding dimension
            dropout: dropout rate
        """
        super().__init__()
        layers = []
        previous_dimension = input_dim

        for hidden_dimension in hidden_dims:
            layers.append(nn.Linear(previous_dimension, hidden_dimension))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            previous_dimension = hidden_dimension

        layers.append(nn.Linear(previous_dimension, output_dim))
        self.multilayer_perceptron = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        encode numeric features

        Args:
            features: numeric features [batch_size, input_dim]

        Returns:
            numeric embeddings [batch_size, output_dim]
        """
        return self.multilayer_perceptron(features)

