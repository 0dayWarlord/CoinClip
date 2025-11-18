#main CoinCLIP model

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from coinclip.models.encoders import ImageEncoder, NumericEncoder, TextEncoder
from coinclip.models.fusion import MultimodalFusion


class CoinCLIPModel(nn.Module):
    """
    multimodal model for memecoin viability assessment

    Combines text, image, and numeric features using a fusion mechanism
    to predict viability/success labels
    """

    def __init__(
        self,
        #encoder configs
        text_encoder_name: str = "bert-base-uncased",
        text_embedding_dim: int = 768,
        image_encoder_name: str = "resnet50",
        image_embedding_dim: int = 2048,
        numeric_input_dim: int = 6,
        numeric_hidden_dims: list[int] = [128, 64],
        numeric_embedding_dim: int = 64,
        #fusion config
        fusion_type: str = "concat",
        fusion_hidden_dim: int = 512,
        fusion_output_dim: int = 256,
        #task config
        num_classes: int = 2,
        #contrastive learning config
        use_contrastive: bool = True,
        contrastive_temperature: float = 0.07,
        #other
        dropout: float = 0.1,
        freeze_text_encoder: bool = False,
        freeze_image_encoder: bool = False,
    ):
        """
        initialize CoinCLIP model

        Args:
            text_encoder_name: text encoder model name
            text_embedding_dim: text embedding dimension
            image_encoder_name: image encoder model name
            image_embedding_dim: image embedding dimension
            numeric_input_dim: numeric feature input dimension
            numeric_hidden_dims: numeric encoder hidden dimensions
            numeric_embedding_dim: numeric embedding dimension
            fusion_type: fusion strategy ('concat', 'attention', 'transformer')
            fusion_hidden_dim: fusion hidden dimension
            fusion_output_dim: fusion output dimension
            num_classes: number of classification classes
            use_contrastive: whether to use contrastive learning
            contrastive_temperature: temperature for contrastive loss
            dropout: dropout rate
            freeze_text_encoder: whether to freeze text encoder
            freeze_image_encoder: whether to freeze image encoder
        """
        super().__init__()

        #modality encoders
        self.text_encoder = TextEncoder(
            model_name=text_encoder_name,
            embedding_dim=text_embedding_dim,
            output_dim=fusion_hidden_dim,
            freeze_backbone=freeze_text_encoder,
        )

        self.image_encoder = ImageEncoder(
            model_name=image_encoder_name,
            embedding_dim=image_embedding_dim,
            output_dim=fusion_hidden_dim,
            freeze_backbone=freeze_image_encoder,
        )

        self.numeric_encoder = NumericEncoder(
            input_dim=numeric_input_dim,
            hidden_dims=numeric_hidden_dims,
            output_dim=fusion_hidden_dim,
            dropout=dropout,
        )

        #store fusion dimensions for partial fusion
        self.fusion_hidden_dim = fusion_hidden_dim
        self.fusion_output_dim = fusion_output_dim
        self.dropout = dropout

        #fusion module (for full 3-modality fusion)
        self.fusion = MultimodalFusion(
            text_dim=fusion_hidden_dim,
            image_dim=fusion_hidden_dim,
            numeric_dim=fusion_hidden_dim,
            output_dim=fusion_output_dim,
            fusion_type=fusion_type,
            hidden_dim=fusion_hidden_dim,
            dropout=dropout,
        )

        #partial fusion MLP for when only 2 modalities are present
        #concatenates 2 modalities and projects to output_dim
        self.partial_fusion = nn.Sequential(
            nn.Linear(fusion_hidden_dim * 2, fusion_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_dim, fusion_output_dim),
        )

        #classification head
        self.classifier = nn.Sequential(
            nn.Linear(fusion_output_dim, fusion_output_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_output_dim // 2, num_classes),
        )

        #contrastive learning
        self.use_contrastive = use_contrastive
        if use_contrastive:
            #projection head for contrastive learning
            self.contrastive_proj = nn.Sequential(
                nn.Linear(fusion_output_dim, fusion_output_dim),
                nn.ReLU(),
                nn.Linear(fusion_output_dim, fusion_output_dim // 2),
            )
            self.contrastive_temperature = contrastive_temperature

    def forward(
        self,
        text: Dict[str, torch.Tensor],
        image: torch.Tensor,
        numeric: torch.Tensor,
        return_embeddings: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        forward pass

        Args:
            text: dictionary with 'input_ids' and 'attention_mask'
            image: image tensor [batch_size, 3, H, W]
            numeric: numeric features [batch_size, num_features]
            return_embeddings: whether to return intermediate embeddings

        Returns:
            dictionary with 'logits' and optionally 'embeddings'
        """
        #encode modalities
        text_embedding = self.text_encoder(
            input_ids=text["input_ids"], attention_mask=text["attention_mask"]
        )
        image_embedding = self.image_encoder(image)
        numeric_embedding = self.numeric_encoder(numeric)

        #fuse modalities
        fused_embedding = self.fusion(text_embedding, image_embedding, numeric_embedding)

        #classification
        logits = self.classifier(fused_embedding)

        output = {"logits": logits}

        if return_embeddings:
            output["text_emb"] = text_embedding
            output["image_emb"] = image_embedding
            output["numeric_emb"] = numeric_embedding
            output["fused_emb"] = fused_embedding

        if self.use_contrastive:
            contrastive_embedding = self.contrastive_proj(fused_embedding)
            output["contrastive_emb"] = contrastive_embedding

        return output

    def encode(
        self,
        text: Optional[Dict[str, torch.Tensor]] = None,
        image: Optional[torch.Tensor] = None,
        numeric: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        encode inputs to fused representation

        Args:
            text: optional text input
            image: optional image input
            numeric: optional numeric input

        Returns:
            fused embeddings
        """
        embeddings = []

        if text is not None:
            text_embedding = self.text_encoder(
                input_ids=text["input_ids"], attention_mask=text["attention_mask"]
            )
            embeddings.append(text_embedding)

        if image is not None:
            image_embedding = self.image_encoder(image)
            embeddings.append(image_embedding)

        if numeric is not None:
            numeric_embedding = self.numeric_encoder(numeric)
            embeddings.append(numeric_embedding)

        if len(embeddings) == 0:
            raise ValueError("At least one modality must be provided")

        #if only one modality, return it directly (already at fusion_hidden_dim)
        if len(embeddings) == 1:
            return embeddings[0]

        #get which modalities are present
        has_text = text is not None
        has_image = image is not None
        has_numeric = numeric is not None

        #if all three modalities are present, use full fusion module
        if has_text and has_image and has_numeric:
            #embeddings are appended in order: text, image, numeric
            text_embedding = embeddings[0]
            image_embedding = embeddings[1]
            numeric_embedding = embeddings[2]
            return self.fusion(text_embedding, image_embedding, numeric_embedding)

        """
        if only two modalities are present, use partial fusion
        concatenate the two available modalities and project to output dimension
        """
        concatenated = torch.cat(embeddings, dim=1)
        return self.partial_fusion(concatenated)

