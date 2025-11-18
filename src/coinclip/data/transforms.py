#data transformation utilities

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T


class TextTransform:
    def __init__(
        self,
        max_length: int = 512,
        tokenizer_name: str = "bert-base-uncased",
        truncation: bool = True,
        padding: bool = True,
    ):
        """
        initialize text transform

        Args:
            max_length: maximum sequence length
            tokenizer_name: huggingFace tokenizer name
            truncation: whether to truncate sequences
            padding: whether to pad sequences
        """
        from transformers import AutoTokenizer

        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.truncation = truncation
        self.padding = padding

    def __call__(self, text: str) -> Dict[str, torch.Tensor]:
        """
        tokenize text

        Args:
            text: input text string

        Returns:
            dictionary with 'input_ids' and 'attention_mask'
        """
        encoded = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=self.truncation,
            padding="max_length" if self.padding else False,
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
        }


class ImageTransform:
    def __init__(
        self,
        image_size: Tuple[int, int] = (224, 224),
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
        augment: bool = False,
    ):
        """
        initialize image transform

        Args:
            image_size: target image size (height, width)
            mean: normalization mean
            std: normalization std
            augment: whether to apply data augmentation
        """
        if augment:
            self.transform = T.Compose(
                [
                    T.Resize(image_size),
                    T.RandomHorizontalFlip(p=0.5),
                    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                    T.ToTensor(),
                    T.Normalize(mean=mean, std=std),
                ]
            )
        else:
            self.transform = T.Compose(
                [
                    T.Resize(image_size),
                    T.ToTensor(),
                    T.Normalize(mean=mean, std=std),
                ]
            )

    def __call__(self, image: Image.Image) -> torch.Tensor:
        """
        transform image

        Args:
            image: PIL image

        Returns:
            transformed image tensor
        """
        if image.mode != "RGB":
            image = image.convert("RGB")
        return self.transform(image)


class NumericTransform:
    def __init__(
        self,
        mean: Optional[np.ndarray] = None,
        standard_deviation: Optional[np.ndarray] = None,
        normalize: bool = True,
    ):
        """
        initialize numeric transform

        Args:
            mean: mean for normalization (computed from data if None)
            standard_deviation: standard deviation for normalization (computed from data if None)
            normalize: whether to normalize features
        """
        self.mean = mean
        self.standard_deviation = standard_deviation
        self.normalize = normalize

    def __call__(self, features: np.ndarray) -> torch.Tensor:
        """
        transform numeric features

        Args:
            features: numeric feature array

        Returns:
            transformed feature tensor
        """
        features = np.array(features, dtype=np.float32)
        if self.normalize and self.mean is not None and self.standard_deviation is not None:
            features = (features - self.mean) / (self.standard_deviation + 1e-8)
        return torch.tensor(features, dtype=torch.float32)

    def fit(self, features_list: List[np.ndarray]) -> None:
        """
        fit transform on data

        Args:
            features_list: list of feature arrays
        """
        if not self.normalize:
            return

        all_features = np.vstack(features_list)
        self.mean = np.mean(all_features, axis=0)
        self.standard_deviation = np.std(all_features, axis=0)

