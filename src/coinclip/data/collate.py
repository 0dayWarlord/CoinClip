#collate function for batching

from typing import Dict, List

import torch
from torch.utils.data.dataloader import default_collate


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    collate function for batching samples

    Args:
        batch: list of samples from dataset

    Returns:
        batched dictionary of tensors
    """
    #separate different modalities
    text_input_ids = []
    text_attention_mask = []
    images = []
    numeric_features = []
    labels = []
    sample_ids = []

    for sample in batch:
        #text
        text_input_ids.append(sample["text"]["input_ids"])
        text_attention_mask.append(sample["text"]["attention_mask"])

        #image
        images.append(sample["image"])

        #numeric
        numeric_features.append(sample["numeric"])

        #label
        labels.append(sample["label"])

        #sample ID (convert to int if string)
        sample_id = sample.get("sample_id", 0)
        if isinstance(sample_id, str):
            #extract numeric part or use hash
            try:
                sample_id = int(''.join(filter(str.isdigit, sample_id)) or 0)
            except:
                sample_id = hash(sample_id) % 1000000  #use hash as fallback
        sample_ids.append(sample_id)

    #stack tensors
    batched = {
        "text": {
            "input_ids": torch.stack(text_input_ids),
            "attention_mask": torch.stack(text_attention_mask),
        },
        "image": torch.stack(images),
        "numeric": torch.stack(numeric_features),
        "label": torch.stack(labels),
        "sample_id": torch.tensor(sample_ids, dtype=torch.long),
    }

    return batched

