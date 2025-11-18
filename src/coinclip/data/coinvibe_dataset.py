#coinVibe dataset implementation

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from coinclip.data.transforms import ImageTransform, NumericTransform, TextTransform


class CoinVibeDataset(Dataset):
    """
    coinVibe dataset for memecoin viability assessment

    Each sample contains:
    - text: description, social posts, whitepaper snippets
    - image: logo, memes
    - numeric: on-chain metrics, market data
    - label: viability/success label
    """

    def __init__(
        self,
        data_dir: Union[str, Path],
        split: str = "train",
        metadata_file: Optional[Union[str, Path]] = None,
        text_transform: Optional[TextTransform] = None,
        image_transform: Optional[ImageTransform] = None,
        numeric_transform: Optional[NumericTransform] = None,
        max_text_length: int = 512,
        image_size: Tuple[int, int] = (224, 224),
        augment: bool = False,
    ):
        """
        initialize CoinVibe dataset

        Args:
            data_dir: root directory containing data
            split: dataset split ('train', 'val', 'test')
            metadata_file: path to metadata CSV/JSON file
            text_transform: optional text transform
            image_transform: optional image transform
            numeric_transform: optional numeric transform
            max_text_length: maximum text sequence length
            image_size: target image size
            augment: whether to apply data augmentation
        """
        self.data_directory = Path(data_dir)
        self.split = split

        #load metadata
        if metadata_file is None:
            metadata_file = self.data_directory / f"{split}_metadata.csv"
        else:
            metadata_file = Path(metadata_file)

        if metadata_file.suffix == ".json":
            with open(metadata_file, "r") as file:
                self.metadata = pd.DataFrame(json.load(file))
        else:
            self.metadata = pd.read_csv(metadata_file)

        #initialize transforms
        self.text_transform = text_transform or TextTransform(
            max_length=max_text_length
        )
        self.image_transform = image_transform or ImageTransform(
            image_size=image_size, augment=augment
        )
        self.numeric_transform = numeric_transform or NumericTransform()

        #fit numeric transform on training data if needed
        if split == "train" and hasattr(self.numeric_transform, 'normalize') and self.numeric_transform.normalize:
            numeric_features = [
                self._load_numeric_features(index) for index in range(len(self.metadata))
            ]
            self.numeric_transform.fit(numeric_features)

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        """
        get a single sample

        Args:
            index: sample index

        Returns:
            dictionary containing all modalities and label
        """
        row = self.metadata.iloc[index]

        #load text
        text = self._load_text(row)
        text_data = self.text_transform(text)

        #load image
        image = self._load_image(row)
        image_data = self.image_transform(image)

        #load numeric features
        numeric_features = self._load_numeric_features(index)
        numeric_data = self.numeric_transform(numeric_features)

        #load label
        label = self._load_label(row)

        return {
            "text": text_data,
            "image": image_data,
            "numeric": numeric_data,
            "label": torch.tensor(label, dtype=torch.long),
            "sample_id": row.get("id", index),
        }

    def _load_text(self, row: pd.Series) -> str:
        """load and concatenate text fields"""
        text_parts = []
        for field in ["description", "social_posts", "whitepaper"]:
            if field in row and pd.notna(row[field]):
                text_parts.append(str(row[field]))
        return " ".join(text_parts) if text_parts else ""

    def _load_image(self, row: pd.Series) -> Image.Image:
        image_path = row.get("image_path", "")
        if not image_path:
            #return a blank image if no path provided
            return Image.new("RGB", (224, 224), color="white")

        full_path = self.data_directory / image_path
        if not full_path.exists():
            return Image.new("RGB", (224, 224), color="white")

        try:
            return Image.open(full_path)
        except Exception:
            return Image.new("RGB", (224, 224), color="white")

    def _load_numeric_features(self, index: int) -> np.ndarray:
        row = self.metadata.iloc[index]
        feature_names = [
            "market_cap",
            "volume_24h",
            "price_change_24h",
            "holders_count",
            "transactions_24h",
            "liquidity",
        ]
        features = []
        for name in feature_names:
            value = row.get(name, 0.0)
            if pd.isna(value):
                value = 0.0
            features.append(float(value))
        return np.array(features, dtype=np.float32)

    def _load_label(self, row: pd.Series) -> int:
        label = row.get("label", row.get("viability", 0))
        if pd.isna(label):
            return 0
        return int(label)

