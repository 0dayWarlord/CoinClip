#multimodal fusion mechanisms

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultimodalFusion(nn.Module):
    """
    multimodal fusion module

    Supports multiple fusion strategies:
    - concat: simple concatenation + MLP
    - attention: cross-modal attention
    - transformer: transformer-based fusion
    """

    def __init__(
        self,
        text_dim: int,
        image_dim: int,
        numeric_dim: int,
        output_dim: int,
        fusion_type: str = "concat",
        hidden_dim: Optional[int] = None,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        """
        initialize fusion module

        Args:
            text_dim: text embedding dimension
            image_dim: image embedding dimension
            numeric_dim: numeric embedding dimension
            output_dim: output dimension
            fusion_type: fusion strategy ('concat', 'attention', 'transformer')
            hidden_dim: hidden dimension for fusion layers
            num_heads: number of attention heads (for attention/transformer)
            num_layers: number of layers (for transformer)
            dropout: dropout rate
        """
        super().__init__()
        self.fusion_type = fusion_type
        total_input_dim = text_dim + image_dim + numeric_dim
        hidden_dim = hidden_dim or output_dim

        if fusion_type == "concat":
            #simple concatenation + MLP
            self.fusion = nn.Sequential(
                nn.Linear(total_input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, output_dim),
            )

        elif fusion_type == "attention":
            #cross-modal attention
            self.text_proj = nn.Linear(text_dim, hidden_dim)
            self.image_proj = nn.Linear(image_dim, hidden_dim)
            self.numeric_proj = nn.Linear(numeric_dim, hidden_dim)

            self.attention = nn.MultiheadAttention(
                hidden_dim, num_heads, dropout=dropout, batch_first=True
            )
            self.norm = nn.LayerNorm(hidden_dim)
            self.output_proj = nn.Linear(hidden_dim * 3, output_dim)

        elif fusion_type == "transformer":
            #transformer-based fusion
            self.text_proj = nn.Linear(text_dim, hidden_dim)
            self.image_proj = nn.Linear(image_dim, hidden_dim)
            self.numeric_proj = nn.Linear(numeric_dim, hidden_dim)

            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                batch_first=True,
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.output_proj = nn.Linear(hidden_dim, output_dim)

        else:
            raise ValueError(f"Unsupported fusion type: {fusion_type}")

    def forward(
        self,
        text_embedding: torch.Tensor,
        image_embedding: torch.Tensor,
        numeric_embedding: torch.Tensor,
    ) -> torch.Tensor:
        """
        fuse multimodal embeddings

        Args:
            text_embedding: text embeddings [batch_size, text_dim]
            image_embedding: image embeddings [batch_size, image_dim]
            numeric_embedding: numeric embeddings [batch_size, numeric_dim]

        Returns:
            fused representation [batch_size, output_dim]
        """
        if self.fusion_type == "concat":
            #concatenate all modalities
            fused = torch.cat([text_embedding, image_embedding, numeric_embedding], dim=1)
            return self.fusion(fused)

        elif self.fusion_type == "attention":
            #project to common dimension
            text_projection = self.text_proj(text_embedding).unsqueeze(1)  #[B, 1, H]
            image_projection = self.image_proj(image_embedding).unsqueeze(1)  #[B, 1, H]
            numeric_projection = self.numeric_proj(numeric_embedding).unsqueeze(1)  #[B, 1, H]

            #stack modalities
            modalities = torch.cat([text_projection, image_projection, numeric_projection], dim=1)  #[B, 3, H]

            #self-attention
            attended, _ = self.attention(modalities, modalities, modalities)
            attended = self.norm(attended + modalities)

            #flatten and project
            fused = attended.view(attended.size(0), -1)
            return self.output_proj(fused)

        elif self.fusion_type == "transformer":
            #project to common dimension
            text_projection = self.text_proj(text_embedding).unsqueeze(1)  #[B, 1, H]
            image_projection = self.image_proj(image_embedding).unsqueeze(1)  #[B, 1, H]
            numeric_projection = self.numeric_proj(numeric_embedding).unsqueeze(1)  #[B, 1, H]

            #stack modalities
            modalities = torch.cat([text_projection, image_projection, numeric_projection], dim=1)  #[B, 3, H]

            #transformer encoding
            encoded = self.transformer(modalities)  #[B, 3, H]

            #use mean pooling or [CLS] token (first position)
            pooled = encoded.mean(dim=1)  #[B, H]
            return self.output_proj(pooled)

        else:
            raise ValueError(f"Unsupported fusion type: {self.fusion_type}")

